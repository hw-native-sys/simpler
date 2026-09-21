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

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "scheduler/scheduler_ready.h"
#include "runtime_types.h"

namespace scheduler_test {

class SchedulerStateBuffer {
public:
    explicit SchedulerStateBuffer(const AicoreSchedulerLayout &layout) :
        base_(std::aligned_alloc(SCHEDULER_STATE_ALIGNMENT, layout.total_size)) {
        EXPECT_NE(base_, nullptr);
        if (base_ != nullptr) EXPECT_TRUE(scheduler_init_data_from_layout(base_, layout));
    }
    ~SchedulerStateBuffer() { std::free(base_); }
    void *base() const { return base_; }

private:
    void *base_{nullptr};
};

template <size_t MaxTaskCount>
class BasicGraphBuffer {
public:
    explicit BasicGraphBuffer(size_t task_count) :
        task_count_(task_count),
        image_(std::make_unique<GraphImage>()) {
        while (capacity_ < std::max<size_t>(task_count, 1))
            capacity_ <<= 1;
        if (capacity_ > kMaxTaskCount) throw std::invalid_argument("test graph exceeds GraphBuffer capacity");
        storage_ = image_->storage.data();
        fanins_ = image_->fanins.data();
        for (size_t task = 0; task < capacity_; ++task) {
            storage_[task].task.task_id = TaskId{static_cast<uint64_t>(task)};
            storage_[task].payload.bind_regions(
                nullptr, nullptr, fanins_ + task * static_cast<size_t>(SCHEDULER_GRAPH_MAX_FANIN)
            );
            if (storage_[task].payload.fanin_data() == nullptr) {
                throw std::logic_error("test graph fanin region must share its contiguous image");
            }
            for (int slot = 0; slot < 3; ++slot)
                storage_[task].task.kernel_id[slot] = INVALID_KERNEL_ID;
        }
    }

    void executable(size_t task, uint8_t subtask_slot, std::vector<int32_t> fanins = {}) {
        ASSERT_LT(task, task_count_);
        ASSERT_LT(subtask_slot, 3);
        ASSERT_LE(fanins.size(), static_cast<size_t>(SCHEDULER_GRAPH_MAX_FANIN));
        storage_[task].task.kernel_id[subtask_slot] = 1;
        storage_[task].payload.fanin_count = static_cast<int32_t>(fanins.size());
        ASSERT_TRUE(fanins.empty() || storage_[task].payload.fanin_data() != nullptr);
        std::copy(fanins.begin(), fanins.end(), storage_[task].payload.fanin_data());
    }

    void mixed(size_t task, uint8_t active_mask) {
        ASSERT_LT(task, task_count_);
        for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if ((active_mask & (1U << subtask_slot)) != 0) storage_[task].task.kernel_id[subtask_slot] = 1;
        }
        storage_[task].payload.fanin_count = 0;
    }

    void predicate(size_t task, uint64_t addr, uint8_t elem_size, uint8_t op, int64_t target = 0) {
        ASSERT_LT(task, task_count_);
        storage_[task].payload.predicate.addr = addr;
        storage_[task].payload.predicate.target = target;
        storage_[task].payload.predicate.elem_size = elem_size;
        storage_[task].payload.predicate.op = static_cast<PredicateOp>(op);
    }

    TaskPayload &payload(size_t task) { return storage_[task].payload; }
    ChipTaskStorage *storage() { return storage_; }

    SchedulerGraphView graph() const {
        return {
            reinterpret_cast<uint64_t>(storage_),
            0,
            task_count_,
            capacity_ - 1,
        };
    }

private:
    static constexpr size_t kMaxTaskCount = MaxTaskCount;
    // Descriptor and payload share one entry, matching the production wire stride.
    struct alignas(64) GraphImage {
        std::array<ChipTaskStorage, kMaxTaskCount> storage{};
        std::array<int32_t, kMaxTaskCount * SCHEDULER_GRAPH_MAX_FANIN> fanins{};
    };

    size_t task_count_;
    size_t capacity_{1};
    std::unique_ptr<GraphImage> image_;
    ChipTaskStorage *storage_{nullptr};
    int32_t *fanins_{nullptr};
};

using GraphBuffer = BasicGraphBuffer<8192>;

struct FixtureStorage {
    explicit FixtureStorage(uint64_t task_count, uint64_t workers = 2) :
        test_contexts(workers),
        test_graphs(workers),
        local_states(workers) {
        EXPECT_TRUE(scheduler_plan_layout(task_count, task_count, 0, &layout));
        scheduler_state = std::make_unique<SchedulerStateBuffer>(layout);
        run_control = scheduler_state_at<SchedulerRunControl>(scheduler_state->base(), layout.run_control_offset);
        contexts = scheduler_state_at<SchedulerWorkerContext>(scheduler_state->base(), layout.worker_contexts_offset);
        run_control->aiv_active_worker_count = workers;
        run_control->scheduler_count = workers;
        for (uint64_t worker = 0; worker < workers; ++worker) {
            SchedulerWorkerContext &context = contexts[worker];
            context.core_type = static_cast<int32_t>(CoreType::AIV);
            context.active = 1;
            context.dispatch_slots_offset = layout.dispatch_slots_offset;
            context.completion_inboxes_offset = layout.completion_inboxes_offset;
            context.task_controls_offset = layout.task_controls_offset;
            context.task_metadata_offset = layout.task_metadata_offset;
            context.ready_inboxes_offset = layout.ready_inboxes_offset;
            context.ready_directory_offset = layout.ready_directory_offset;
            context.trace_cells_offset = layout.trace_cells_offset;
            context.worker_contexts_offset = layout.worker_contexts_offset;
            context.callable_addresses_offset = layout.callable_addresses_offset;
            context.gang_coordinator_offset = layout.gang_coordinator_offset;
            context.gang_cohorts_offset = layout.gang_cohorts_offset;
            context.gang_participants_offset = layout.gang_participants_offset;
            context.gang_commands_offset = layout.gang_commands_offset;
            context.dispatch_payload_offset =
                layout.dispatch_payloads_offset + worker * SCHEDULER_PENDING_SLOT_COUNT * sizeof(DispatchPayload);
            context.graph_task_count = task_count;
            context.runtime_worker_count = workers;
            context.worker_index = worker;
            context.scheduler_index = worker;
        }
        metadata = scheduler_state_at<SchedulerTaskMetadata>(scheduler_state->base(), layout.task_metadata_offset);
        callable_addresses = scheduler_state_at<uint64_t>(scheduler_state->base(), layout.callable_addresses_offset);
        callable_addresses[1] = UINT64_C(0x1000);
        for (uint64_t task = 0; task < task_count; ++task) {
            metadata[task].kernel_ids[0] = 1;
            metadata[task].kernel_ids[1] = UINT16_MAX;
            metadata[task].kernel_ids[2] = UINT16_MAX;
            metadata[task].active_mask = 1;
            metadata[task].logical_block_num = 1;
            metadata[task].total_required_subtasks = 1;
            metadata[task].flags = SCHEDULER_TASK_EXECUTABLE;
        }
    }

    // Queue/dispatch tests configure synthetic wire contexts directly. Snapshot
    // their immutable fields into the local state used by the operation under test.
    SchedulerLocalState *local_context(const SchedulerWorkerContext *context, SchedulerLocalState *local = nullptr) {
        if (local == nullptr) local = &test_contexts[context - contexts];
        auto &config = local->config;
        config.dispatch_payloads_offset = static_cast<uint32_t>(layout.dispatch_payloads_offset);
        config.shared_context = context;
        auto &graph = test_graphs[context - contexts];
        if (graph.task_count != context->graph_task_count) graph.task_count = context->graph_task_count;
        config.graph = &graph;
        config.task_controls_offset = context->task_controls_offset;
        config.task_metadata_offset = context->task_metadata_offset;
        config.ready_inboxes_offset = context->ready_inboxes_offset;
        config.ready_directory_offset = context->ready_directory_offset;
        config.trace_cells_offset = context->trace_cells_offset;
        config.activity_buffers_offset = context->activity_buffers_offset;
        config.worker_contexts_offset = context->worker_contexts_offset;
        config.callable_addresses_offset = context->callable_addresses_offset;
        config.gang_coordinator_offset = context->gang_coordinator_offset;
        config.runtime_worker_count = context->runtime_worker_count;
        config.scheduler_index = context->scheduler_index;
        config.scheduler_count = context->scheduler_count;
        config.self_lane = 0;
        config.scheduler_lane = UINT8_MAX;
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            config.worker_ids[lane] = context->cluster_worker_ids[lane];
            if (context->cluster_worker_ids[lane] == context->worker_index) config.self_lane = lane;
        }
        config.worker_ids[config.self_lane] = context->worker_index;
        if (context->is_scheduler != 0) config.scheduler_lane = config.self_lane;
        return local;
    }
    std::vector<SchedulerLocalState> test_contexts;
    std::vector<SchedulerGraphView> test_graphs;

    AicoreSchedulerLayout layout{};
    std::unique_ptr<SchedulerStateBuffer> scheduler_state;
    SchedulerRunControl *run_control{nullptr};
    SchedulerWorkerContext *contexts{nullptr};
    SchedulerTaskMetadata *metadata{nullptr};
    uint64_t *callable_addresses{nullptr};
    std::vector<SchedulerLocalState> local_states;
    SchedulerLocalState *owner_states{local_states.data()};
    SchedulerLocalState scheduler_local_state{};
};

}  // namespace scheduler_test
