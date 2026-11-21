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
#include <memory>
#include <type_traits>
#include <vector>

#include "callable.h"
#include "orchestrator.h"
#include "runtime_core.h"
#include "scheduler/scheduler_context.h"

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
enum class ProgressPath { UnsafeEarly, GlobalProducerHintDisabled, LocalDummyGuard };

MixedKernels kernels(KernelKind kind) {
    MixedKernels result;
    if (kind != KernelKind::Aiv) result.aic_kernel_id = 0;
    if (kind != KernelKind::Aic) result.aiv0_kernel_id = 0;
    if (kind == KernelKind::Mix) result.aiv1_kernel_id = 0;
    return result;
}

class EarlyDispatchDeferredCompletion : public testing::Test {
protected:
    static constexpr int kWindow = 256;
    static constexpr uint64_t kHeapBytes = 1 << 20;
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
    OrchestratorLayout orch_layout{};
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
        orch_layout = OrchestratorState::reserve_layout(runtime_arena, windows, 1024);
        const auto sched_layout = SchedulerState::reserve_layout(runtime_arena, 1024);
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
        KernelKind kind, int16_t blocks = 1, bool early = false, const TaskId *dependencies = nullptr,
        int dependency_count = 1, const twait::Declaration *declaration = nullptr
    ) {
        CoreTaskArgs args;
        args.launch_spec.set_block_num(blocks);
        args.set_allow_early_resolve(early);
        const std::vector<DepFlags> flags(dependency_count, DEP_WAIT);
        if (dependencies != nullptr) args.set_dependencies_with_kinds(dependencies, flags.data(), dependency_count);
        if (declaration != nullptr) args.set_blocking_wait(*declaration);
        const auto result = orch->submit_task(kernels(kind), args);
        EXPECT_FALSE(orch->fatal);
        EXPECT_TRUE(result.task_id().is_valid());
        return result.task_id();
    }

    TaskId submit_guard(TaskId producer) {
        CoreTaskArgs args;
        args.set_allow_early_resolve(false);
        const DepFlags flags = DEP_WAIT;
        args.set_dependencies_with_kinds(&producer, &flags, 1);
        const auto result = orch->submit_dummy_task(args);
        EXPECT_FALSE(orch->fatal);
        EXPECT_TRUE(result.task_id().is_valid());
        return result.task_id();
    }

    template <typename Entry>
    int complete_dummy_batch(int (ChipReadyQueue::*pop_batch)(Entry *, int)) {
        Entry batch[8]{};
        ChipTaskSlotState *slots[8]{};
        const int count = (scheduler->dummy_ready_queue.*pop_batch)(batch, 8);
        for (int i = 0; i < count; ++i) {
            if constexpr (std::is_pointer_v<Entry>) {
                slots[i] = batch[i];
            } else {
                slots[i] = batch[i].ptr();
            }
        }
        for (int i = 0; i < count; ++i) {
#if SIMPLER_SCHED_PROFILING
            (void)scheduler->on_task_complete(*slots[i], 0);
#else
            (void)scheduler->on_task_complete(*slots[i]);
#endif
        }
        for (int i = 0; i < count; ++i) {
#if SIMPLER_SCHED_PROFILING
            (void)scheduler->on_task_release(*slots[i], 0);
#else
            scheduler->on_task_release(*slots[i]);
#endif
        }
        return count;
    }

    int complete_dummies() { return complete_dummy_batch(&ChipReadyQueue::pop_batch); }

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

    twait::Declaration declare(TaskId producer, TaskId barrier) {
        twait::Declaration declaration;
        declaration.runtime = orch.get();
        declaration.run = orch->twait_run_generation;
        declaration.rank = orch->twait_rank;
        declaration.event = 17;
        declaration.epoch = 23;
        declaration.condition = 1;
        declaration.completion_implies_condition = true;
        declaration.all_contributors_declared = true;
        declaration.count = 1;
        declaration.producers[0] = {
            producer,
            orch->wait_reach[producer.ring()]
                            [shared.header->rings[producer.ring()].get_slot_by_task_id(producer.local_id())]
                                .seq,
            orch->twait_rank,
            static_cast<uint32_t>(slot(producer)->logical_block_num),
            barrier,
            orch->wait_reach[barrier.ring()]
                            [shared.header->rings[barrier.ring()].get_slot_by_task_id(barrier.local_id())]
                                .seq
        };
        return declaration;
    }

    void defer_producer(TaskId producer, volatile uint32_t *notification) {
        dispatch();
        ASSERT_EQ(core(0).running_slot_state, slot(producer));
        ack(0);
        MixAdmissionTestPeer::register_counter(*context, 0, notification);
        conditions[0] = MAKE_FIN_VALUE(core(0).running_reg_task_id);
        ASSERT_EQ(poll(), 0);
        ASSERT_EQ(core(0).running_slot_state, nullptr);
        ASSERT_EQ(poll_async(), 0);
        ASSERT_EQ(scheduler->async_wait_list.count.load(), 1);
        ASSERT_TRUE(scheduler->async_wait_list.entries[0].normal_done);
        ASSERT_EQ(scheduler->async_wait_list.entries[0].waiting_completion_count, 1);
        ASSERT_EQ(slot(producer)->task_state.load(), CHIP_TASK_PENDING);
    }

    void reject(const CoreTaskArgs &args, const char *reason, TaskId producer = TaskId::invalid()) {
        const auto ring = orch->current_ring_id();
        const auto head = orch->rings[ring].task_allocator.task_head();
        const auto next_mapping = orch->tensor_map.next_entry_idx;
        const auto scope_size = orch->scope_tasks_size;
        const auto fanin_top = orch->rings[ring].fanin_pool.top;
        const auto proof_count = orch->twait_proofs.count;
        const auto sequence = orch->submit_seq;
        const auto producer_claims = producer.is_valid() ? slot(producer)->fanout_count : 0;
        uint32_t ready[NUM_RESOURCE_SHAPES]{}, early[NUM_RESOURCE_SHAPES]{};
        for (int shape = 0; shape < NUM_RESOURCE_SHAPES; ++shape) {
            ready[shape] = scheduler->ready_queues[shape].size();
            early[shape] = scheduler->early_dispatch_queues[shape].size();
        }
        const auto dummy = scheduler->dummy_ready_queue.size();
        const auto sync = scheduler->early_sync_start_queue.size();
        const auto result = orch->submit_task(kernels(KernelKind::Aiv), args);
        ASSERT_FALSE(result.task_id().is_valid());
        ASSERT_TRUE(orch->fatal);
        ASSERT_NE(orch->twait_last_error, nullptr);
        EXPECT_STREQ(orch->twait_last_error, reason);
        EXPECT_EQ(shared.header->orch_error_code.load(), SIMPLER_ERROR_INVALID_ARGS);
        EXPECT_EQ(orch->rings[ring].task_allocator.task_head(), head);
        EXPECT_EQ(orch->tensor_map.next_entry_idx, next_mapping);
        EXPECT_EQ(orch->scope_tasks_size, scope_size);
        EXPECT_EQ(orch->rings[ring].fanin_pool.top, fanin_top);
        EXPECT_EQ(orch->twait_proofs.count, proof_count);
        EXPECT_EQ(orch->submit_seq, sequence);
        if (producer.is_valid()) EXPECT_EQ(slot(producer)->fanout_count, producer_claims);
        for (int shape = 0; shape < NUM_RESOURCE_SHAPES; ++shape) {
            EXPECT_EQ(scheduler->ready_queues[shape].size(), ready[shape]);
            EXPECT_EQ(scheduler->early_dispatch_queues[shape].size(), early[shape]);
        }
        EXPECT_EQ(scheduler->dummy_ready_queue.size(), dummy);
        EXPECT_EQ(scheduler->early_sync_start_queue.size(), sync);
    }

    void expect_no_occupancy(TaskId task, EarlyDispatchState early_state = EARLY_DISPATCH_NONE) {
        auto *state = slot(task);
        for (int id = 0; id < core_count; ++id) {
            EXPECT_NE(core(id).running_slot_state, state);
            EXPECT_NE(core(id).pending_slot_state, state);
        }
        EXPECT_EQ(state->next_block_idx.load(), 0);
        EXPECT_EQ(state->payload->published_block_count.load(), 0);
        EXPECT_EQ(state->payload->early_dispatch_state.load(), early_state);
        EXPECT_EQ(state->payload->early_sync_drain_state.load(), EARLY_SYNC_DRAIN_NONE);
        for (auto &mask : state->payload->staged_core_mask)
            EXPECT_EQ(mask.load(), 0U);
        EXPECT_EQ(scheduler->early_sync_start_queue.size(), 0U);
        EXPECT_EQ(MixAdmissionTestPeer::drain_pending(*context), 0);
        EXPECT_EQ(MixAdmissionTestPeer::drain_task(*context), nullptr);
    }
};

class DeferredCompletionProgress : public EarlyDispatchDeferredCompletion,
                                   public testing::WithParamInterface<ProgressPath> {};

TEST_P(DeferredCompletionProgress, LateNotifierNeedsAnUnoccupiedExecutionSlot) {
    const auto path = GetParam();
    const bool producer_early = path != ProgressPath::GlobalProducerHintDisabled;
    const bool guarded = path == ProgressPath::LocalDummyGuard;
    const bool progress_expected = path != ProgressPath::UnsafeEarly;
    alignas(64) volatile uint32_t notification = 0;
    const TaskId producer = submit(KernelKind::Aic, 1, producer_early);
    dispatch();
    ASSERT_EQ(core(0).running_slot_state, slot(producer));
    ack(0);
    MixAdmissionTestPeer::register_counter(*context, 0, &notification);
    conditions[0] = MAKE_FIN_VALUE(core(0).running_reg_task_id);
    ASSERT_EQ(poll(), 0);
    ASSERT_EQ(core(0).running_slot_state, nullptr);
    ASSERT_EQ(poll_async(), 0);
    const auto &waits = scheduler->async_wait_list;
    ASSERT_EQ(waits.count.load(std::memory_order_relaxed), 1);
    const auto &deferred = waits.entries[0];
    ASSERT_EQ(deferred.task_token, producer);
    ASSERT_TRUE(deferred.normal_done);
    ASSERT_EQ(deferred.waiting_completion_count, 1);
    ASSERT_EQ(slot(producer)->completed_subtasks.load(), 1);
    ASSERT_EQ(slot(producer)->task_state.load(), CHIP_TASK_PENDING);
    ASSERT_EQ(slot(producer)->has_dispatch_propagated(), producer_early);

    const TaskId guard = guarded ? submit_guard(producer) : TaskId::invalid();
    const TaskId dependency = guarded ? guard : producer;
    const auto declaration = declare(producer, dependency);
    const TaskId waiter = submit(KernelKind::Aiv, 2, true, &dependency, 1, progress_expected ? &declaration : nullptr);
    ASSERT_TRUE(waiter.is_valid());
    ASSERT_FALSE(orch->fatal);
    ASSERT_EQ(slot(waiter)->fanin_count, 2);
    ASSERT_EQ(slot(waiter)->fanin_refcount.load(), 1);
    ASSERT_TRUE(slot(waiter)->task_attrs.allow_early_resolve());
    dispatch();
    if (guarded) {
        expect_no_occupancy(guard);
        ASSERT_FALSE(slot(guard)->task_attrs.allow_early_resolve());
        ASSERT_FALSE(slot(guard)->has_dispatch_propagated());
        ASSERT_EQ(scheduler->dummy_ready_queue.size(), 0U);
        ASSERT_EQ(complete_dummies(), 0);
    }

    const TaskId enabling_task = submit(KernelKind::Aic);
    const TaskId notifier = submit(KernelKind::Aiv, 1, false, &enabling_task);
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
        if (guarded) {
            expect_no_occupancy(guard, EARLY_DISPATCH_DISPATCHED);
            ASSERT_EQ(slot(waiter)->fanin_refcount.load(), 1);
            ASSERT_EQ(scheduler->dummy_ready_queue.size(), 1U);
            ASSERT_EQ(complete_dummies(), 1);
            ASSERT_GE(slot(guard)->task_state.load(), CHIP_TASK_COMPLETED);
            expect_no_occupancy(guard, EARLY_DISPATCH_DISPATCHED);
            ASSERT_EQ(slot(waiter)->fanin_refcount.load(), slot(waiter)->fanin_count);
        }
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
    ExistingPolicies, DeferredCompletionProgress,
    testing::Values(ProgressPath::UnsafeEarly, ProgressPath::GlobalProducerHintDisabled, ProgressPath::LocalDummyGuard)
);

TEST_F(EarlyDispatchDeferredCompletion, LocalGuardKeepsOtherConsumersAndWaiterSuccessorsEarlyEligible) {
    const TaskId producer = submit(KernelKind::Aic, 1, true);
    dispatch();
    ack(0);
    const TaskId other_consumer = submit(KernelKind::Aiv, 1, false, &producer);
    const TaskId guard = submit_guard(producer);
    const TaskId waiter = submit(KernelKind::Aiv, 1, true, &guard);
    const TaskId successor = submit(KernelKind::Aic, 1, false, &waiter);
    dispatch();
    ASSERT_TRUE(slot(producer)->task_attrs.allow_early_resolve());
    ASSERT_EQ(core(1).running_slot_state, slot(other_consumer));
    ASSERT_NE(MixAdmissionTestPeer::running_gate(*context, 1), 0U);
    expect_no_occupancy(guard);
    expect_no_occupancy(waiter);
    finish(producer);
    ASSERT_EQ(complete_dummies(), 1);
    dispatch();
    ASSERT_TRUE(slot(waiter)->task_attrs.allow_early_resolve());
    ASSERT_TRUE(slot(waiter)->has_dispatch_propagated());
    ASSERT_EQ(core(2).running_slot_state, slot(waiter));
    ASSERT_EQ(core(0).running_slot_state, slot(successor));
    ASSERT_NE(MixAdmissionTestPeer::running_gate(*context, 0), 0U);
    finish(other_consumer);
    finish(waiter);
    finish(successor);
    orch->end_scope();
}

TEST_F(EarlyDispatchDeferredCompletion, CompletedGuardAndScopedSlotReuseDoNotInventAStaleBarrier) {
    const TaskId producer = submit(KernelKind::Aic, 1, true);
    const TaskId guard = submit_guard(producer);
    auto *guard_slot = slot(guard);
    dispatch();
    finish(producer);
    ASSERT_EQ(complete_dummies(), 1);
    ASSERT_GE(slot(guard)->task_state.load(), CHIP_TASK_COMPLETED);
    expect_no_occupancy(guard, EARLY_DISPATCH_DISPATCHED);
    const TaskId waiter = submit(KernelKind::Aiv, 1, true, &guard);
    ASSERT_EQ(slot(waiter)->fanin_refcount.load(), slot(waiter)->fanin_count);
    dispatch();
    finish(waiter);
    orch->end_scope();
    for (int task = 3; task < kWindow; ++task) {
        orch->begin_scope();
        const TaskId ordinary = submit(KernelKind::Aic);
        dispatch();
        finish(ordinary);
        orch->end_scope();
    }
    ASSERT_EQ(orch->rings[0].task_allocator.task_head(), kWindow);
    ASSERT_EQ(orch->rings[0].task_allocator.task_tail(), kWindow);
    orch->begin_scope();
    const TaskId next_producer = submit(KernelKind::Aic, 1, true);
    dispatch();
    ack(0);
    const TaskId ordinary = submit(KernelKind::Aiv, 1, false, &next_producer);
    ASSERT_EQ(slot(ordinary), guard_slot);
    ASSERT_EQ(slot(ordinary)->payload->early_dispatch_blocked, 0);
    dispatch();
    ASSERT_EQ(core(1).running_slot_state, slot(ordinary));
    ASSERT_NE(MixAdmissionTestPeer::running_gate(*context, 1), 0U);
    finish(next_producer);
    finish(ordinary);
    orch->end_scope();
}

TEST_F(EarlyDispatchDeferredCompletion, ReducedDummyWaitEdgeKeepsItsEarlyAdmissionBlocker) {
    const TaskId producer = submit(KernelKind::Aic, 1, true);
    const TaskId guard = submit_guard(producer);
    const TaskId intermediate = submit(KernelKind::Aiv, 1, true, &guard);
    const TaskId dependencies[] = {guard, intermediate};
    const TaskId waiter = submit(KernelKind::Aiv, 1, true, dependencies, 2);
    ASSERT_EQ(slot(waiter)->payload->fanin_actual_count, 2);
    ASSERT_EQ(slot(waiter)->payload->fanin_wait_count, 1);
    ASSERT_EQ(slot(waiter)->payload->early_dispatch_blocked, 1);
    ASSERT_EQ(slot(waiter)->payload->early_dispatch_target(), 2);
    bool guard_present = false;
    for_each_fanin_slot_state(*slot(waiter)->payload, [&](ChipTaskSlotState *input, DepFlags flags) {
        if (input == slot(guard)) {
            guard_present = true;
            EXPECT_FALSE(dep_has_wait(flags));
        }
    });
    ASSERT_TRUE(guard_present);
    dispatch();
    expect_no_occupancy(guard);
    expect_no_occupancy(waiter);
    finish(producer);
    ASSERT_EQ(complete_dummies(), 1);
    dispatch();
    ASSERT_EQ(core(1).running_slot_state, slot(intermediate));
    ASSERT_TRUE(slot(intermediate)->has_dispatch_propagated());
    ASSERT_EQ(slot(waiter)->payload->dispatch_fanin.load(), 1);
    expect_no_occupancy(waiter);
    finish(intermediate);
    ASSERT_EQ(slot(waiter)->fanin_refcount.load(), slot(waiter)->fanin_count);
    dispatch();
    finish(waiter);
    orch->end_scope();
}

class BlockingAdmission : public EarlyDispatchDeferredCompletion {};

TEST_F(BlockingAdmission, ValidWaitEarlyDeferredIsRejectedBeforePublication) {
    alignas(64) volatile uint32_t notification = 0;
    const TaskId producer = submit(KernelKind::Aic, 1, true);
    defer_producer(producer, &notification);
    ASSERT_FALSE(HasFatalFailure());
    auto declaration = declare(producer, producer);
    CoreTaskArgs args;
    args.set_allow_early_resolve(true);
    const DepFlags flags = DEP_WAIT;
    args.set_dependencies_with_kinds(&producer, &flags, 1);
    args.set_blocking_wait(declaration);
    const uint32_t shape[] = {16};
    const TensorCreateInfo output(shape, 1);
    args.add_output(output);
    reject(args, "no completion-before-occupancy certificate", producer);
}

TEST_F(BlockingAdmission, RetentionIsNotACompletionCertificate) {
    const auto producer = submit(KernelKind::Aic);
    auto declaration = declare(producer, producer);
    CoreTaskArgs args;
    const DepFlags flags = DEP_RETAIN;
    args.set_dependencies_with_kinds(&producer, &flags, 1);
    args.set_blocking_wait(declaration);
    reject(args, "missing DEP_WAIT completion barrier", producer);
}

TEST_F(BlockingAdmission, EmptyDeclarationIsRejected) {
    twait::Declaration declaration;
    declaration.runtime = orch.get();
    declaration.run = orch->twait_run_generation;
    CoreTaskArgs args;
    args.set_blocking_wait(declaration);
    reject(args, "missing event producer declaration");
}

TEST_F(BlockingAdmission, FutureProducerIsRejectedWithoutIndexing) {
    const auto producer = submit(KernelKind::Aic);
    auto declaration = declare(producer, producer);
    declaration.producers[0].task = TaskId::make(0, 1000000);
    CoreTaskArgs args;
    args.set_blocking_wait(declaration);
    reject(args, "future producer");
}

TEST_F(BlockingAdmission, WrongGenerationIsRejected) {
    const auto producer = submit(KernelKind::Aic);
    auto declaration = declare(producer, producer);
    ++declaration.producers[0].generation;
    CoreTaskArgs args;
    args.set_blocking_wait(declaration);
    reject(args, "wrong producer generation", producer);
}

TEST_F(BlockingAdmission, WrongRunIsRejected) {
    const auto producer = submit(KernelKind::Aic);
    auto declaration = declare(producer, producer);
    ++declaration.run;
    CoreTaskArgs args;
    args.set_blocking_wait(declaration);
    reject(args, "wrong run or rank", producer);
}

TEST_F(BlockingAdmission, WrongRankIsRejected) {
    const auto producer = submit(KernelKind::Aic);
    auto declaration = declare(producer, producer);
    ++declaration.producers[0].rank;
    CoreTaskArgs args;
    args.set_blocking_wait(declaration);
    reject(args, "remote completion certificate unsupported", producer);
}

TEST_F(BlockingAdmission, RetainOnlyTransitiveAncestryDoesNotProveAWaitBarrier) {
    const auto producer = submit(KernelKind::Aic, 1, true);
    CoreTaskArgs barrier_args;
    const DepFlags retain = DEP_RETAIN;
    barrier_args.set_dependencies_with_kinds(&producer, &retain, 1);
    const auto barrier = orch->submit_task(kernels(KernelKind::Aiv), barrier_args).task_id();
    ASSERT_TRUE(barrier.is_valid());
    auto declaration = declare(producer, barrier);
    CoreTaskArgs args;
    const DepFlags wait = DEP_WAIT;
    args.set_dependencies_with_kinds(&barrier, &wait, 1);
    args.set_blocking_wait(declaration);
    reject(args, "unproved producer-to-barrier WAIT path", producer);
}

TEST_F(BlockingAdmission, AllBlocksMustBeDeclared) {
    const auto producer = submit(KernelKind::Aic, 2, true);
    auto declaration = declare(producer, producer);
    declaration.producers[0].blocks = 1;
    CoreTaskArgs args;
    args.set_blocking_wait(declaration);
    reject(args, "incomplete producer block set", producer);
}

TEST_F(BlockingAdmission, InvalidRingIsRejectedBeforeAnyArrayAccess) {
    const auto producer = submit(KernelKind::Aic);
    auto declaration = declare(producer, producer);
    declaration.producers[0].task = TaskId::make(255, 0);
    CoreTaskArgs args;
    args.set_blocking_wait(declaration);
    reject(args, "invalid producer identity", producer);
}

TEST_F(BlockingAdmission, RetiredProducerIsNotConfusedWithAReusedSlot) {
    const auto producer = submit(KernelKind::Aic);
    auto declaration = declare(producer, producer);
    dispatch();
    finish(producer);
    orch->end_scope();
    for (int index = 1; index <= kWindow; ++index) {
        orch->begin_scope();
        const auto ordinary = submit(KernelKind::Aic);
        dispatch();
        finish(ordinary);
        orch->end_scope();
    }
    orch->begin_scope();
    CoreTaskArgs args;
    args.set_blocking_wait(declaration);
    reject(args, "stale producer");
}

class BlockingAcceptance : public EarlyDispatchDeferredCompletion {};

TEST_F(BlockingAcceptance, CompletedProducerRequiresNoResidentBarrier) {
    const auto producer = submit(KernelKind::Aic, 1, true);
    dispatch();
    finish(producer);
    auto declaration = declare(producer, producer);
    const auto waiter = submit(KernelKind::Aiv, 1, true, nullptr, 0, &declaration);
    dispatch();
    finish(waiter);
    orch->end_scope();
}

TEST_F(BlockingAcceptance, DeclarationIsOwnedAndRetiredOnScopeAndRunReuse) {
    const auto producer = submit(KernelKind::Aic, 1, true);
    dispatch();
    finish(producer);
    auto declaration = declare(producer, producer);
    const auto waiter = submit(KernelKind::Aiv, 1, true, nullptr, 0, &declaration);
    ASSERT_TRUE(waiter.is_valid());
    const twait::RetainedDeclaration *stored = nullptr;
    for (const auto &entry : orch->twait_declarations)
        if (entry.task == waiter) stored = &entry;
    ASSERT_NE(stored, nullptr);
    ASSERT_EQ(stored->event.event, 17U);
    declaration.event = 999;
    declaration.producers[0].task = TaskId::invalid();
    ASSERT_EQ(stored->event.event, 17U);
    ASSERT_EQ(stored->producers[0].task, producer);
    dispatch();
    finish(waiter);
    orch->end_scope();
    for (const auto &entry : orch->twait_declarations)
        ASSERT_EQ(entry.waiter.run, 0U);
    const auto old_run = declaration.run;
    uint64_t heap_sizes[CHIP_MAX_RING_DEPTH], windows[CHIP_MAX_RING_DEPTH];
    for (int ring = 0; ring < CHIP_MAX_RING_DEPTH; ++ring) {
        heap_sizes[ring] = kHeapBytes;
        windows[ring] = kWindow;
    }
    ASSERT_TRUE(orch->reset_for_reuse(orch_layout, shared.sm_base, heap.data(), heap_sizes, windows));
    ASSERT_NE(orch->twait_run_generation, old_run);
    CoreTaskArgs stale_args;
    stale_args.set_blocking_wait(declaration);
    reject(stale_args, "wrong run or rank");
}

TEST_F(BlockingAcceptance, ReducedUnflaggedCertificateKeepsWorkerCapacityFree) {
    alignas(64) volatile uint32_t notification = 0;
    const auto producer = submit(KernelKind::Aic, 1, true);
    defer_producer(producer, &notification);
    const auto guard = submit_guard(producer);
    const auto intermediate = submit(KernelKind::Aiv, 1, true, &guard);
    const TaskId dependencies[] = {guard, intermediate};
    const auto declaration = declare(producer, guard);
    const auto waiter = submit(KernelKind::Aiv, 1, true, dependencies, 2, &declaration);
    ASSERT_TRUE(waiter.is_valid());
    ASSERT_FALSE(orch->fatal);
    ASSERT_EQ(slot(waiter)->payload->early_dispatch_blocked, 1);
    dispatch();
    expect_no_occupancy(waiter);
    const auto notifier = submit(KernelKind::Aiv);
    dispatch();
    ASSERT_EQ(core(1).running_slot_state, slot(notifier));
    ack(1);
    notification = 1;
    conditions[1] = MAKE_FIN_VALUE(core(1).running_reg_task_id);
    ASSERT_EQ(poll(), 1);
    ASSERT_EQ(poll_async(), 1);
    ASSERT_EQ(complete_dummies(), 1);
    dispatch();
    expect_no_occupancy(waiter);
    finish(intermediate);
    dispatch();
    finish(waiter);
    orch->end_scope();
}

class ColdCycles : public EarlyDispatchDeferredCompletion {
protected:
    twait::Snapshot cycle(bool cross_rank, const twait::Declaration &declaration) {
        twait::Snapshot snapshot;
        snapshot.count = cross_rank ? 2 : 1;
        const auto prospective = TaskId::make(0, orch->rings[0].task_allocator.task_head());
        const twait::Identity local{orch->twait_run_generation, 0, orch->submit_seq, TaskId::to_uint64(prospective), 0};
        const twait::Identity remote{orch->twait_run_generation, 1, 3, 401, 0};
        snapshot.records[0].waiter = local;
        snapshot.records[0].event = {orch->twait_run_generation, 0, 17, 23, 1};
        snapshot.records[0].producer = {
            orch->twait_run_generation, declaration.producers[0].rank, declaration.producers[0].generation,
            TaskId::to_uint64(declaration.producers[0].task), 0
        };
        snapshot.records[0].complete = true;
        snapshot.records[0].alternative_count = 1;
        snapshot.records[0].alternatives[0] = {orch->twait_run_generation, cross_rank ? 1U : 0U, 8, 1, 0, 0, true, true,
                                               cross_rank ? remote : local};
        if (cross_rank) {
            snapshot.records[1].waiter = remote;
            snapshot.records[1].event = {orch->twait_run_generation, 1, 18, 24, 1};
            snapshot.records[1].producer = {orch->twait_run_generation, 0, 10, 102, 0};
            snapshot.records[1].complete = true;
            snapshot.records[1].alternative_count = 1;
            snapshot.records[1].alternatives[0] = {orch->twait_run_generation, 0, 9, 1, 0, 0, true, true, local};
        }
        return snapshot;
    }

    void check_cycle(bool cross_rank) {
        const auto producer = submit(KernelKind::Aic, 1, true);
        auto declaration = declare(producer, producer);
        if (cross_rank) declaration.producers[0].rank = 1;
        auto snapshot = cycle(cross_rank, declaration);
        const auto witness = twait::diagnose(snapshot);
        ASSERT_EQ(witness.diagnosis, twait::Diagnosis::Cycle);
        ASSERT_EQ(witness.count, snapshot.count);
        for (uint32_t index = 0; index < witness.count; ++index) {
            const auto &hop = witness.hops[index];
            EXPECT_TRUE(hop.waiter == snapshot.records[index].waiter);
            EXPECT_EQ(hop.event.event, snapshot.records[index].event.event);
            EXPECT_TRUE(hop.producer == snapshot.records[index].producer);
            EXPECT_TRUE(hop.core_slot.owner == witness.hops[(index + 1) % witness.count].waiter);
        }
        auto broken = snapshot;
        broken.records[0].alternative_count = 0;
        EXPECT_NE(twait::diagnose(broken).diagnosis, twait::Diagnosis::Cycle);
        auto free_capacity = snapshot;
        auto &record = free_capacity.records[0];
        record.alternative_count = 2;
        record.alternatives[1] = record.alternatives[0];
        record.alternatives[1].core = 1;
        record.alternatives[1].occupied = false;
        EXPECT_NE(twait::diagnose(free_capacity).diagnosis, twait::Diagnosis::Cycle);
        auto pending_only = snapshot;
        pending_only.records[0].alternatives[0].executing = false;
        EXPECT_NE(twait::diagnose(pending_only).diagnosis, twait::Diagnosis::Cycle);
        auto absent_owner = snapshot;
        ++absent_owner.records[0].alternatives[0].owner.generation;
        EXPECT_NE(twait::diagnose(absent_owner).diagnosis, twait::Diagnosis::Cycle);
        declaration.snapshot = &snapshot;
        CoreTaskArgs args;
        const DepFlags flags = DEP_WAIT;
        args.set_dependencies_with_kinds(&producer, &flags, 1);
        args.set_blocking_wait(declaration);
        reject(
            args,
            cross_rank ? "remote completion certificate unsupported" : "no completion-before-occupancy certificate",
            producer
        );
        ASSERT_FALSE(HasFatalFailure());
        ASSERT_EQ(orch->twait_last_witness.diagnosis, twait::Diagnosis::Cycle);
        ASSERT_EQ(orch->twait_last_witness.count, snapshot.count);
        const auto &hop = orch->twait_last_witness.hops[0];
        EXPECT_TRUE(hop.waiter == snapshot.records[0].waiter);
        EXPECT_TRUE(hop.producer == snapshot.records[0].producer);
        EXPECT_EQ(hop.core_slot.rank, snapshot.records[0].alternatives[0].rank);
        EXPECT_EQ(hop.event.event, declaration.event);
    }
};

TEST_F(ColdCycles, SameCore) { check_cycle(false); }
TEST_F(ColdCycles, CrossDeviceRecords) { check_cycle(true); }

TEST_F(EarlyDispatchDeferredCompletion, OrdinaryMixedSpmdPredicateAndTimingRemainUnannotated) {
    configure_clusters(2);
    CoreTaskArgs args;
    args.launch_spec.set_block_num(2);
    args.set_task_timing_slot(0);
    CoreTaskPredicate predicate;
    args.set_predicate(predicate);
    ASSERT_FALSE(args.has_blocking_wait());
    const auto result = orch->submit_task(kernels(KernelKind::Mix), args);
    ASSERT_FALSE(orch->fatal);
    ASSERT_TRUE(result.task_id().is_valid());
    dispatch();
    ASSERT_EQ(slot(result.task_id())->logical_block_num, 2);
    finish(result.task_id());
    args.set_blocking_wait(twait::Declaration{});
    args.clear();
    ASSERT_FALSE(args.has_blocking_wait());
    orch->end_scope();
}

class LongSpanProof : public EarlyDispatchDeferredCompletion,
                      public testing::WithParamInterface<std::tuple<int, bool, int>> {
protected:
    twait::ProofHandle handle(TaskId task) {
        const auto sequence =
            orch->wait_reach[task.ring()][shared.header->rings[task.ring()].get_slot_by_task_id(task.local_id())].seq;
        const auto position = orch->twait_proofs.find(orch->twait_run_generation, task, sequence);
        EXPECT_GE(position, 0);
        return {orch->twait_run_generation, static_cast<uint32_t>(position)};
    }

    TaskId proof_task() {
        CoreTaskArgs args;
        args.capture_wait_proof();
        args.set_allow_early_resolve(true);
        const auto result = orch->submit_task(kernels(KernelKind::Aic), args);
        EXPECT_TRUE(result.task_id().is_valid());
        EXPECT_FALSE(orch->fatal);
        return result.task_id();
    }
};

TEST_P(LongSpanProof, AllContributorsBeforeOccupancy) {
    const auto distance = std::get<0>(GetParam());
    const auto sparse = std::get<1>(GetParam());
    const auto mutation = std::get<2>(GetParam());
    alignas(64) volatile uint32_t first_notification = 0;
    alignas(64) volatile uint32_t second_notification = 0;
    TaskId roots[2];
    roots[0] = proof_task();
    defer_producer(roots[0], &first_notification);
    ASSERT_FALSE(HasFatalFailure());
    roots[1] = proof_task();
    dispatch();
    ASSERT_EQ(core(0).running_slot_state, slot(roots[1]));
    ack(0);
    MixAdmissionTestPeer::register_counter(*context, 0, &second_notification);
    conditions[0] = MAKE_FIN_VALUE(core(0).running_reg_task_id);
    ASSERT_EQ(poll(), 0);
    ASSERT_EQ(poll_async(), 0);
    ASSERT_EQ(slot(roots[1])->task_state.load(), CHIP_TASK_PENDING);
    TaskId barrier = TaskId::invalid();
    while (orch->submit_seq <= static_cast<uint64_t>(distance)) {
        CoreTaskArgs args;
        TaskId dependencies[2] = {roots[0], roots[1]};
        DepFlags flags[2] = {DEP_WAIT, mutation == 2 ? DEP_RETAIN : DEP_WAIT};
        if (mutation == 1) flags[1] = DEP_NONE;
        int count = 2;
        if (barrier.is_valid()) {
            dependencies[0] = barrier;
            flags[0] = DEP_WAIT;
            count = 1;
        }
        const bool filler = sparse && orch->submit_seq < static_cast<uint64_t>(distance);
        if (!filler) {
            args.capture_wait_proof();
            args.set_dependencies_with_kinds(dependencies, flags, count);
        }
        const auto result = orch->submit_dummy_task(args);
        ASSERT_TRUE(result.task_id().is_valid());
        ASSERT_FALSE(orch->fatal);
        if (filler) ASSERT_EQ(complete_dummies(), 1);
        else barrier = result.task_id();
    }
    ASSERT_TRUE(barrier.is_valid());
    auto declaration = declare(roots[0], barrier);
    declaration.count = 2;
    declaration.producers[0].proof = handle(barrier);
    declaration.producers[1] = declare(roots[1], barrier).producers[0];
    declaration.producers[1].proof = handle(barrier);
    ASSERT_EQ(
        declaration.producers[0].barrier_generation - declaration.producers[0].generation,
        static_cast<uint64_t>(distance)
    );
    if (mutation == 3) ++declaration.producers[1].proof.run;
    if (mutation == 4) ++declaration.producers[1].generation;
    if (mutation == 5) declaration.all_contributors_declared = false;
    CoreTaskArgs args;
    const DepFlags wait = DEP_WAIT;
    args.set_dependencies_with_kinds(&barrier, &wait, 1);
    args.set_allow_early_resolve(true);
    args.set_blocking_wait(declaration);
    if (mutation != 0) {
        const char *reason = mutation == 4 ? "wrong producer generation" :
                             mutation == 5 ? "missing event producer declaration" :
                                             "unproved producer-to-barrier WAIT path";
        reject(args, reason, roots[0]);
        return;
    }
    const auto waiter = orch->submit_task(kernels(KernelKind::Aiv), args).task_id();
    ASSERT_TRUE(waiter.is_valid());
    ASSERT_FALSE(orch->fatal);
    declaration = twait::Declaration{};
    dispatch();
    expect_no_occupancy(waiter);
    const auto notifier = submit(KernelKind::Aiv);
    dispatch();
    ASSERT_EQ(core(1).running_slot_state, slot(notifier));
    ack(1);
    first_notification = 1;
    ASSERT_EQ(poll_async(), 1);
    ASSERT_EQ(complete_dummies(), 0);
    dispatch();
    expect_no_occupancy(waiter);
    second_notification = 1;
    conditions[1] = MAKE_FIN_VALUE(core(1).running_reg_task_id);
    ASSERT_EQ(poll(), 1);
    ASSERT_EQ(poll_async(), 1);
    int released = 0;
    while (const auto batch = complete_dummies())
        released += batch;
    ASSERT_GT(released, 0);
    dispatch();
    finish(waiter);
    orch->end_scope();
}

INSTANTIATE_TEST_SUITE_P(
    SpanBoundaries, LongSpanProof,
    testing::Combine(
        testing::Values(WAIT_REACH_WINDOW, WAIT_REACH_WINDOW + 1, 2 * WAIT_REACH_WINDOW + 1), testing::Bool(),
        testing::Values(0, 1, 2, 3, 4, 5)
    )
);

class ProofLifecycle : public EarlyDispatchDeferredCompletion {};

TEST_F(ProofLifecycle, ExhaustionRejectsWithoutPublicationAndOrdinaryTasksDoNoCapture) {
    for (uint32_t index = 0; index < twait::max_proof_nodes; ++index) {
        CoreTaskArgs args;
        args.capture_wait_proof();
        const auto task = orch->submit_dummy_task(args).task_id();
        ASSERT_TRUE(task.is_valid());
        ASSERT_EQ(complete_dummies(), 1);
        orch->end_scope();
        orch->begin_scope();
    }
    ASSERT_EQ(orch->twait_proofs.count, twait::max_proof_nodes);
    const auto ordinary = submit(KernelKind::Aiv);
    dispatch();
    finish(ordinary);
    ASSERT_EQ(orch->twait_proofs.count, twait::max_proof_nodes);
    CoreTaskArgs args;
    args.capture_wait_proof();
    reject(args, "WAIT proof storage full");
}

TEST_F(ProofLifecycle, SharedAncestorsSurviveRetirementAndResetInvalidatesHandles) {
    CoreTaskArgs args;
    args.capture_wait_proof();
    const auto root = orch->submit_dummy_task(args).task_id();
    ASSERT_TRUE(root.is_valid());
    const auto old_run = orch->twait_run_generation;
    const auto old_node = orch->twait_proofs.nodes[0];
    ASSERT_EQ(complete_dummies(), 1);
    orch->end_scope();
    for (int index = 0; index <= kWindow; ++index) {
        orch->begin_scope();
        const auto ordinary = orch->submit_dummy_task(CoreTaskArgs{}).task_id();
        ASSERT_TRUE(ordinary.is_valid());
        ASSERT_EQ(complete_dummies(), 1);
        orch->end_scope();
    }
    ASSERT_EQ(orch->twait_proofs.nodes[0].task, old_node.task);
    ASSERT_EQ(orch->twait_proofs.nodes[0].generation, old_node.generation);
    uint64_t heaps[CHIP_MAX_RING_DEPTH], windows[CHIP_MAX_RING_DEPTH];
    for (int ring = 0; ring < CHIP_MAX_RING_DEPTH; ++ring) {
        heaps[ring] = kHeapBytes;
        windows[ring] = kWindow;
    }
    ASSERT_TRUE(orch->reset_for_reuse(orch_layout, shared.sm_base, heap.data(), heaps, windows));
    ASSERT_NE(orch->twait_run_generation, old_run);
    ASSERT_EQ(orch->twait_proofs.count, 0U);
    ASSERT_FALSE(orch->twait_proofs.proves(
        {old_run, 0}, orch->twait_run_generation, root, old_node.generation, root, old_node.generation
    ));
    args.clear();
    ASSERT_FALSE(args.captures_wait_proof());
}

}  // namespace
