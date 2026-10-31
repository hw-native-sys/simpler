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

#pragma once

#include <gtest/gtest.h>
#include <array>
#include <thread>
#include <atomic>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "callable.h"
#include "scheduler/scheduler_context.h"
#include "test_task_id.h"

struct ChipTaskStorage;

void __attribute__((weak)) reg_store_release(volatile uint32_t *address, uint32_t value) { *address = value; }

namespace mix_lane_test {
constexpr int capacity = 8;
LaneMask mask_of(int bits) { return {bool(bits & 1), bool(bits & 2), bool(bits & 4)}; }
template <typename Slot, typename = void>
struct TaskStorage {
    using Type = ChipTaskStorage;
};

template <typename Slot>
struct TaskStorage<Slot, std::void_t<decltype(std::declval<Slot>().task), decltype(std::declval<Slot>().payload)>> {
    struct Type {
        TaskDescriptor task{};
        TaskPayload payload{};
        Slot slot{};
        Type() {
            slot.task = &task;
            slot.payload = &payload;
        }
    };
};

struct TestTask {
    TaskStorage<ChipTaskSlotState>::Type storage{};
    TaskDescriptor &descriptor = storage.task;
    TaskPayload &payload = storage.payload;
    ChipTaskSlotState &slot = storage.slot;
    void init(int bits, int blocks, int id = 0) {
        slot.active_mask = mask_of(bits);
        slot.logical_block_num = blocks;
        slot.total_required_subtasks = blocks * slot.active_mask.count();
        descriptor.task_id = simpler::ut::test_task_id(id);
        for (int i = 0; i < 3; ++i)
            descriptor.kernel_id[i] = (bits & (1 << i)) ? 0 : INVALID_KERNEL_ID;
    }
};
void init_queue(ChipReadyQueue &q, ChipReadyQueueSlot *slots) {
    q.capacity = capacity;
    q.mask = capacity - 1;
    q.slots = slots;
    q.enqueue_pos.store(0);
    q.dequeue_pos.store(0);
    q.mixed_lanes.reset();
    for (int i = 0; i < capacity; ++i) {
        slots[i].sequence.store(i);
        slots[i].slot_value = CapturedReadySlot{};
    }
}
}  // namespace mix_lane_test

using mix_lane_test::capacity;
using mix_lane_test::init_queue;
using mix_lane_test::mask_of;
using mix_lane_test::TestTask;

class SchedulerContextTestPeer {
public:
    static void
    bind(SchedulerContext &c, SchedulerState &s, uint64_t *callables, uint64_t *regs, size_t words, bool pending) {
        c.sched_ = &s;
        c.active_sched_threads_ = 1;
        c.func_id_to_addr_ = callables;
        c.func_id_to_addr_count_ = 1;
        c.completed_.store(false);
        c.drain_state_.sync_start_pending.store(0);
        auto &tracker = c.core_trackers_[0];
        tracker.init(1);
        tracker.set_cluster(0, 0, 1, 2);
        for (int i = 0; i < 3; ++i) {
            auto &core = c.core_exec_states_[i];
            core = {};
            core.reg_addr = reinterpret_cast<uint64_t>(regs + i * words);
            core.running_reg_task_id = pending ? 0 : AICPU_TASK_INVALID;
            core.pending_reg_task_id = AICPU_TASK_INVALID;
            if (pending) tracker.change_core_state(i);
        }
    }
    static void leaf(SchedulerContext &c, ResourceShape shape, bool pending, bool early, LaneMask &observed) {
        const auto phase = pending ? CoreTracker::DispatchPhase::PENDING : CoreTracker::DispatchPhase::IDLE;
        if (early) c.early_dispatch_shape(0, shape, phase, observed);
        else {
            bool drain = false, progress = false, pushed = false;
            c.dispatch_shape(
                0, c.sched_->ready_queues, shape, phase, c.core_trackers_[0], drain, progress, pushed,
                SchedulerContext::MixPriority::Ready, observed
            );
        }
    }
    static void sync_leaf(SchedulerContext &c, LaneMask &observed) {
        bool stop = false, progress = false, pushed = false;
        c.dispatch_shape(
            0, c.sched_->ready_sync_queues, ResourceShape::AIV, CoreTracker::DispatchPhase::IDLE, c.core_trackers_[0],
            stop, progress, pushed, SchedulerContext::MixPriority::ReadySync, observed
        );
    }
    static void ready(SchedulerContext &c, bool pmu) {
        bool progress = false, pushed = false;
        c.dispatch_ready_tasks(0, c.core_trackers_[0], pmu, progress, pushed);
    }
    static LaneMask allowed(SchedulerContext &c, int priority, LaneMask &observed) {
        return c.unreserved_mix_lanes(static_cast<SchedulerContext::MixPriority>(priority), observed);
    }
    static int drain_pending(SchedulerContext &c) { return c.drain_state_.sync_start_pending.load(); }
    static void block_lane(SchedulerContext &c, int lane, ChipTaskSlotState *blocker) {
        c.core_trackers_[0].change_core_state(lane);
        c.core_trackers_[0].set_pending_occupied(lane);
        c.core_exec_states_[lane].running_slot_state = blocker;
        c.core_exec_states_[lane].running_reg_task_id = 0;
    }
    static void drain(SchedulerContext &c, int state) { c.drain_state_.sync_start_pending.store(state); }
    static ChipTaskSlotState *placed(SchedulerContext &c, int lane, bool pending) {
        const auto &core = c.core_exec_states_[lane];
        return pending ? core.pending_slot_state : core.running_slot_state;
    }
    static bool requeue(
        SchedulerContext &c, ChipReadyQueue &q, const CapturedReadySlot *values, int count, ResourceShape shape,
        int thread_idx = 0
    ) {
        return c.requeue_ready_tasks(thread_idx, q, values, count, shape);
    }
    static bool stopped(SchedulerContext &c) { return c.completed_.load(); }
};

class LanePriority : public testing::TestWithParam<std::tuple<int, bool, bool>> {
protected:
    std::unique_ptr<SchedulerState> sched = std::make_unique<SchedulerState>();
    std::unique_ptr<SchedulerContext> context = std::make_unique<SchedulerContext>();
    SharedMemoryHeader header{};
    ChipReadyQueueSlot cells[10][capacity]{};
    TestTask mixed, pure;
    std::vector<uint8_t> callable_bytes = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, nullptr, 0);
    uint64_t callables[1]{};
    std::vector<uint64_t> registers;
    void SetUp() override {
        sched->sm_header = &header;
        int index = 0;
        for (int shape = 0; shape < 3; ++shape) {
            init_queue(sched->ready_sync_queues[shape], cells[index++]);
            init_queue(sched->ready_queues[shape], cells[index++]);
            init_queue(sched->early_dispatch_queues[shape], cells[index++]);
        }
        init_queue(sched->early_sync_start_queue, cells[index]);
        auto *callable = reinterpret_cast<CoreCallable *>(callable_bytes.data());
        callable->set_resolved_addr(1);
        callables[0] = reinterpret_cast<uint64_t>(callable);
        const size_t words = (reg_offset(RegId::COND) + sizeof(uint64_t)) / sizeof(uint64_t);
        registers.resize(3 * words);
        SchedulerContextTestPeer::bind(*context, *sched, callables, registers.data(), words, std::get<1>(GetParam()));
        mixed.init(std::get<0>(GetParam()), 1, 0);
    }
    void enqueue_mix(bool early = false) {
        auto &q = early ? sched->early_dispatch_queues[2] : sched->ready_queues[2];
        ASSERT_TRUE(q.push_mix(&mixed.slot, mixed.descriptor.task_id));
    }
};
