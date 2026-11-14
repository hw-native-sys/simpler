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
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "scheduler/scheduler_ready.h"
#include "task_catalog_plan.h"
#include "prepared_execution_plan.h"
#include "runtime_types.h"

namespace scheduler_test {

inline void protocol_noop_kernel(__gm__ int64_t *) {}

class SchedulerStateBuffer {
public:
    explicit SchedulerStateBuffer(const AicoreSchedulerLayout &layout) :
        allocation_(
            std::aligned_alloc(
                SCHEDULER_STATE_BASE_ALIGNMENT, (layout.total_size + SCHEDULER_STATE_BASE_ALIGNMENT - 1) /
                                                    SCHEDULER_STATE_BASE_ALIGNMENT * SCHEDULER_STATE_BASE_ALIGNMENT
            )
        ),
        base_(allocation_) {
        // Shared records require the 128-byte atomic-region alignment.
        EXPECT_NE(base_, nullptr);
        if (base_ != nullptr)
            EXPECT_TRUE(scheduler_init_data_from_layout(base_, layout, reinterpret_cast<uint64_t>(base_)));
    }
    ~SchedulerStateBuffer() { std::free(allocation_); }
    void *base() const { return base_; }

private:
    void *allocation_{nullptr};
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
            storage_[task].task.task_id = TaskId::make_global(static_cast<int32_t>(task));
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

    void mixed(size_t task, LaneMask active_mask) {
        ASSERT_LT(task, task_count_);
        for (uint8_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if (active_mask.has(static_cast<SubtaskSlot>(subtask_slot)))
                storage_[task].task.kernel_id[subtask_slot] = 1;
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
        return scheduler_ordinary_graph_view(
            reinterpret_cast<uint64_t>(storage_), static_cast<uint32_t>(task_count_), capacity_ - 1
        );
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

// Entries the fixture's own callable table can address. Published as each
// worker context's count, so a case that wants the length to be the bound
// overrides that context rather than resizing this.
inline constexpr uint64_t kFixtureCallableCount = 1024;

struct FixtureStorage {
    explicit FixtureStorage(
        uint64_t task_count, uint64_t workers = 2, uint64_t traces_per_task = 1,
        const SchedulerSpmdRequirements &spmd = {}, bool protocol_profiling = false,
        const std::function<void(AicoreSchedulerLayout &)> &extend_layout = {}
    ) :
        test_profiles(workers),
        test_contexts(workers),
        test_graphs(workers),
        local_states(workers) {
        for (auto &local : test_contexts)
            for (auto &slot : local.slots)
                scheduler_initialize_free_slot(&slot);
        for (auto &slot : scheduler_local_state.slots)
            scheduler_initialize_free_slot(&slot);
        EXPECT_TRUE(scheduler_plan_layout(
            task_count, task_count, 0, &layout, false, task_count * traces_per_task, spmd, protocol_profiling
        ));
        prepared_capacity = layout.prepared.execution_count;
        if (extend_layout) extend_layout(layout);
        // Reserve backing once. The final metadata later determines the two
        // actual capacities; together they can never exceed task_count. A
        // one-entry second catalog covers the largest split-alignment padding.
        catalog_begin = layout.total_size;
        std::vector<SchedulerTaskMetadata> upper_bound(task_count);
        for (auto &task : upper_bound) {
            task.flags = SCHEDULER_TASK_EXECUTABLE;
            task.active_mask = LaneMask::aic();
        }
        if (task_count > 1) upper_bound.back().active_mask = LaneMask::aiv0();
        SchedulerTaskCatalogPlan reserve{};
        if (!scheduler_plan_task_catalog(upper_bound, catalog_clusters(), catalog_begin, &reserve))
            throw std::logic_error("cannot reserve fixture task catalog");
        layout.total_size = reserve.total_size;
        scheduler_state = std::make_unique<SchedulerStateBuffer>(layout);
        run_control = scheduler_state_at<SchedulerRunControl>(scheduler_state->base(), layout.run_control_offset);
        contexts = scheduler_state_at<SchedulerWorkerContext>(scheduler_state->base(), layout.worker_contexts_offset);
        run_control->aiv_active_worker_count = workers;
        run_control->scheduler_count = workers;
        for (uint64_t worker = 0; worker < workers; ++worker) {
            SchedulerWorkerContext &context = contexts[worker];
            context.core_type =
                static_cast<int32_t>(worker % PLATFORM_CORES_PER_BLOCKDIM == 0 ? CoreType::AIC : CoreType::AIV);
            context.task_controls_offset = layout.task_controls_offset;
            context.task_metadata_offset = layout.task_metadata_offset;

            context.trace_cells_offset = layout.trace_cells_offset;
            context.worker_contexts_offset = layout.worker_contexts_offset;
            context.callable_addresses_address = reinterpret_cast<uint64_t>(callable_addresses);
            context.callable_addresses_count = kFixtureCallableCount;
            context.graph_summary_offset = layout.graph_summary_offset;
            context.graph_task_count = task_count;
            context.original_task_count = task_count;
            context.graph_view_version = SCHEDULER_GRAPH_VIEW_VERSION;
            context.graph_view_bytes = sizeof(SchedulerGraphView);
            context.scheduler_state_bytes = layout.total_size;
            context.runtime_worker_count = workers;
            context.worker_index = worker;
            context.scheduler_index = worker;
            // One cluster per three workers, each lane ranked within its own
            // type, matching what the AICPU assigns during core discovery.
            const uint64_t lane = worker % PLATFORM_CORES_PER_BLOCKDIM;
            const uint64_t cluster = worker / PLATFORM_CORES_PER_BLOCKDIM;
            context.cluster_count = (workers + PLATFORM_CORES_PER_BLOCKDIM - 1) / PLATFORM_CORES_PER_BLOCKDIM;
            context.cluster_index = cluster;
            context.type_rank =
                static_cast<int32_t>(lane == 0 ? cluster : cluster * PLATFORM_AIV_CORES_PER_BLOCKDIM + (lane - 1));
        }
        metadata = scheduler_state_at<SchedulerTaskMetadata>(scheduler_state->base(), layout.task_metadata_offset);
        callable_addresses[1] =
            protocol_profiling ? reinterpret_cast<uint64_t>(&protocol_noop_kernel) : UINT64_C(0x1000);
        for (uint64_t task = 0; task < task_count; ++task) {
            metadata[task].kernel_ids[0] = 1;
            metadata[task].kernel_ids[1] = UINT16_MAX;
            metadata[task].kernel_ids[2] = UINT16_MAX;
            metadata[task].active_mask = LaneMask::aic();
            metadata[task].trace_index_base = task;
            metadata[task].flags = SCHEDULER_TASK_EXECUTABLE;
            metadata[task].logical_block_num = 1;
            auto *controls =
                scheduler_state_at<SchedulerTaskControl>(scheduler_state->base(), layout.task_controls_offset);
            controls[task].completion_state = scheduler_completion_initial_state(1);
        }
        if (layout.prepared_layout_offset != 0) {
            auto initial_layout = layout;
            initial_layout.prepared.execution_count = task_count;
            scheduler_state_at<SchedulerPreparedLayout>(scheduler_state->base(), layout.prepared_layout_offset)
                ->execution_count = task_count;
            EXPECT_TRUE(scheduler_init_prepared_execution_records(scheduler_state->base(), initial_layout, 0));
        }
    }

    bool prepare_calls(const SchedulerGraphView &graph) {
        for (auto &view : test_graphs)
            view = graph;
        if (calls_prepared || layout.prepared_layout_offset == 0) return true;
        for (uint64_t ref = 0; ref < layout.task_count; ++ref) {
            const auto &meta = metadata[ref];
            if (!scheduler_task_is_executable(meta.flags) || scheduler_task_is_inline(meta.flags)) continue;
            SchedulerTaskPayloadArguments arguments{};
            if (scheduler_parse_task_payload_arguments(graph, ref, &arguments) != SchedulerGraphResult::OK)
                return false;
            uint64_t functions[SUBTASK_SLOT_COUNT]{};
            for (uint8_t lane = 0; lane < SUBTASK_SLOT_COUNT; ++lane) {
                if (!meta.active_mask.has(static_cast<SubtaskSlot>(lane))) continue;
                if (meta.kernel_ids[lane] >= kFixtureCallableCount) return false;
                functions[lane] = callable_addresses[meta.kernel_ids[lane]];
            }
            if (!scheduler_prepare_host_calls(
                    scheduler_state->base(), layout, reinterpret_cast<uint64_t>(scheduler_state->base()), ref,
                    functions, reinterpret_cast<uint64_t>(arguments.tensors), arguments.scalars, arguments.tensor_count,
                    arguments.scalar_count
                ))
                return false;
        }
        calls_prepared = true;
        return true;
    }

    uint32_t catalog_clusters() const {
        return std::max<uint32_t>(
            1, static_cast<uint32_t>(
                   (test_contexts.size() + PLATFORM_CORES_PER_BLOCKDIM - 1) / PLATFORM_CORES_PER_BLOCKDIM
               )
        );
    }

    // The catalog and execution identities are frozen after metadata is finalized.
    // This initialization precedes every execution claim in the allocation.
    void prepare_catalog() {
        ASSERT_EQ(task_catalog_offset, 0u);
        uint64_t execution_count = 0;
        for (uint64_t task = 0; task < layout.task_count; ++task) {
            const auto &meta = metadata[task];
            if (scheduler_task_is_executable(meta.flags) && !scheduler_task_is_inline(meta.flags))
                execution_count += meta.logical_block_num * meta.active_mask.count();
        }
        ASSERT_LE(execution_count, prepared_capacity);
        layout.prepared.execution_count = execution_count;
        if (layout.prepared_layout_offset != 0)
            scheduler_state_at<SchedulerPreparedLayout>(scheduler_state->base(), layout.prepared_layout_offset)
                ->execution_count = execution_count;
        const auto epoch =
            scheduler_state_at<SchedulerGraphSummary>(scheduler_state->base(), layout.graph_summary_offset)->run_epoch;
        ASSERT_TRUE(scheduler_init_prepared_execution_records(scheduler_state->base(), layout, epoch));
        auto *controls = scheduler_state_at<SchedulerTaskControl>(scheduler_state->base(), layout.task_controls_offset);
        for (uint64_t task = 0; task < layout.task_count; ++task) {
            const auto &meta = metadata[task];
            const uint32_t total = scheduler_task_is_executable(meta.flags) && !scheduler_task_is_inline(meta.flags) ?
                                       meta.logical_block_num * meta.active_mask.count() :
                                       1;
            controls[task].completion_state = scheduler_completion_initial_state(total);
        }
        const std::vector<SchedulerTaskMetadata> tasks(metadata, metadata + layout.task_count);
        SchedulerTaskCatalogPlan plan{};
        ASSERT_TRUE(scheduler_plan_task_catalog(tasks, catalog_clusters(), catalog_begin, &plan));
        ASSERT_GE(plan.header_offset, catalog_begin);
        ASSERT_LE(plan.total_size, layout.total_size);
        scheduler_init_task_catalog(scheduler_state->base(), plan);
        task_catalog_offset = plan.header_offset;
        scheduler_state_at<SchedulerGraphSummary>(scheduler_state->base(), layout.graph_summary_offset)
            ->task_catalog_offset = task_catalog_offset;
        for (auto &local : test_contexts)
            local.config.task_catalog_offset = task_catalog_offset;
        for (auto &local : local_states)
            local.config.task_catalog_offset = task_catalog_offset;
        scheduler_local_state.config.task_catalog_offset = task_catalog_offset;
    }

    // Dispatch tests configure synthetic wire contexts directly. Snapshot
    // their immutable fields into the local state used by the operation under test.
    SchedulerLocalState *local_context(const SchedulerWorkerContext *context, SchedulerLocalState *local = nullptr) {
        if (local == nullptr) local = &test_contexts[context - contexts];
        if (local->profiling == nullptr) local->profiling = &test_profiles[context - contexts];
        auto &config = local->config;
        config.shared_context = context;
        config.prepared =
            layout.prepared_layout_offset == 0 ?
                nullptr :
                scheduler_state_at<SchedulerPreparedLayout>(scheduler_state->base(), layout.prepared_layout_offset);
        auto &graph = test_graphs[context - contexts];
        if (graph.task_count != context->graph_task_count) graph.task_count = context->graph_task_count;
        config.graph = &graph;
        config.task_controls_offset = context->task_controls_offset;
        config.task_metadata_offset = context->task_metadata_offset;
        config.task_catalog_offset = task_catalog_offset;
        local->profiling->trace_cells_offset = context->trace_cells_offset;
        local->profiling->activity_buffers_offset = context->activity_buffers_offset;
        config.worker_contexts_offset = context->worker_contexts_offset;
        config.callable_addresses_address = context->callable_addresses_address;
        config.callable_addresses_count = static_cast<uint32_t>(context->callable_addresses_count);
        config.graph_summary_offset = context->graph_summary_offset;
        config.runtime_worker_count = context->runtime_worker_count;
        config.scheduler_index = context->scheduler_index;
        config.scheduler_count = context->scheduler_count;
        config.self_lane = 0;
        config.self_core_type = scheduler_core_type_index(context->core_type);
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            config.worker_ids[lane] = context->cluster_worker_ids[lane];
            if (context->cluster_worker_ids[lane] == context->worker_index) config.self_lane = lane;
        }
        config.worker_ids[config.self_lane] = context->worker_index;
        config.cluster_count = static_cast<uint16_t>(context->cluster_count);
        return local;
    }
    std::vector<SchedulerLocalProfilingState> test_profiles;
    std::vector<SchedulerLocalState> test_contexts;
    std::vector<SchedulerGraphView> test_graphs;

    AicoreSchedulerLayout layout{};
    uint64_t prepared_capacity{0};
    bool calls_prepared{false};
    uint64_t catalog_begin{0};
    uint64_t task_catalog_offset{0};
    std::unique_ptr<SchedulerStateBuffer> scheduler_state;
    SchedulerRunControl *run_control{nullptr};
    SchedulerWorkerContext *contexts{nullptr};
    SchedulerTaskMetadata *metadata{nullptr};
    // The callable's registration-owned entry table. It is not part of the
    // scheduler state, so the fixture owns one and hands every worker context
    // its address, exactly as a bind hands over the address of the block a
    // registration retained.
    std::unique_ptr<std::array<uint64_t, kFixtureCallableCount>> callable_table{
        std::make_unique<std::array<uint64_t, kFixtureCallableCount>>()
    };
    uint64_t *callable_addresses{callable_table->data()};

    std::vector<SchedulerLocalState> local_states;
    SchedulerLocalState *owner_states{local_states.data()};
    SchedulerLocalProfilingState scheduler_profile{};
    SchedulerLocalState scheduler_local_state{&scheduler_profile};
};

}  // namespace scheduler_test
