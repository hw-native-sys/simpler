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
#include <cstdint>
#include <utility>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_completion.h"
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_protocol_execute.h"
#include "test_hbg_owned_execution.h"

namespace scheduler_test::early_dispatch {

struct Shape {
    uint16_t blocks{1};
    LaneMask mask{LaneMask::aic()};
    bool sync_start{false};
    uint8_t extra_flags{0};
};

struct Execution {
    int64_t task{SCHEDULER_TASK_REF_INVALID};
    uint32_t block{0};
    uint32_t blocks{0};
    uint32_t worker{0};
    uint32_t slot{0};
    uint64_t function{0};
};

inline uint64_t &kernel_run_count() {
    static uint64_t count = 0;
    return count;
}

inline void counting_kernel(__gm__ int64_t *) { ++kernel_run_count(); }

inline uint8_t first_lane(LaneMask mask) {
    for (uint8_t slot = 0; slot < 3; ++slot)
        if (mask.has(static_cast<SubtaskSlot>(slot))) return slot;
    return 0;
}

// Host-driven resident scheduler fixture: every transition is an explicit
// production helper call, so the test itself fixes the event order it asserts.
struct Fixture {
    static constexpr uint64_t kMaxWorkers = 8;

    explicit Fixture(const std::vector<Shape> &shapes, uint64_t epoch = 37, uint64_t workers = 6) :
        storage(shapes.size(), workers, max_blocks(shapes), requirements(shapes, workers)),
        graph(shapes.size()),
        shape_list(shapes),
        worker_count(workers) {
        configure(epoch);
    }

    static uint64_t max_blocks(const std::vector<Shape> &shapes) {
        uint64_t maximum = 1;
        for (const auto &shape : shapes)
            maximum = std::max<uint64_t>(maximum, shape.blocks);
        return maximum;
    }

    static SchedulerSpmdRequirements requirements(const std::vector<Shape> &shapes, uint64_t workers) {
        SchedulerSpmdRequirements requirements{
            workers, (workers + PLATFORM_CORES_PER_BLOCKDIM - 1) / PLATFORM_CORES_PER_BLOCKDIM, 0, 0, 0
        };
        for (const auto &shape : shapes) {
            const bool mix = shape.mask.count() > 1;
            if (shape.blocks > 1 && shape.sync_start) ++requirements.cooperative_task_count;
            if (mix) ++requirements.mix_task_count;
            if (mix && shape.blocks > 1 && shape.sync_start) ++requirements.sync_mix_task_count;
        }
        return requirements;
    }

    // One full host initialization of the same allocation, matching what a
    // launch performs. Reused verbatim for repeated runs under a new epoch.
    void configure(uint64_t epoch) {
        run_epoch = epoch;
        void *base = storage.scheduler_state->base();
        ASSERT_TRUE(scheduler_init_data_from_layout(base, storage.layout, reinterpret_cast<uint64_t>(base)));
        storage.run_control->active_worker_count = worker_count;
        for (uint64_t worker = 0; worker < worker_count; ++worker) {
            SchedulerWorkerContext &context = storage.contexts[worker];
            context.core_type =
                static_cast<int32_t>(worker % PLATFORM_CORES_PER_BLOCKDIM == 0 ? CoreType::AIC : CoreType::AIV);
            context.task_controls_offset = storage.layout.task_controls_offset;
            context.task_metadata_offset = storage.layout.task_metadata_offset;
            context.done_notifications_offset = storage.layout.done_notifications_offset;
            context.completion_counters_offset = storage.layout.completion_counters_offset;
            context.trace_cells_offset = storage.layout.trace_cells_offset;
            context.worker_contexts_offset = storage.layout.worker_contexts_offset;
            context.graph_summary_offset = storage.layout.graph_summary_offset;
            context.graph_task_count = shape_list.size();
            context.original_task_count = shape_list.size();
            context.scheduler_state_base_address = reinterpret_cast<uint64_t>(base);
            context.scheduler_state_bytes = storage.layout.total_size;
            context.runtime_worker_count = worker_count;
            context.worker_index = worker;
            const uint64_t lane = worker % PLATFORM_CORES_PER_BLOCKDIM;
            const uint64_t cluster = worker / PLATFORM_CORES_PER_BLOCKDIM;
            context.cluster_count = (worker_count + PLATFORM_CORES_PER_BLOCKDIM - 1) / PLATFORM_CORES_PER_BLOCKDIM;
            context.cluster_index = cluster;
            context.type_rank =
                static_cast<int32_t>(lane == 0 ? cluster : cluster * PLATFORM_AIV_CORES_PER_BLOCKDIM + (lane - 1));
            for (uint32_t member = 0; member < PLATFORM_CORES_PER_BLOCKDIM; ++member)
                context.cluster_worker_ids[member] = cluster * PLATFORM_CORES_PER_BLOCKDIM + member;
        }
        uint32_t trace = 0;
        for (uint64_t task = 0; task < shape_list.size(); ++task) {
            const auto shape = shape_list[task];
            auto &metadata = storage.metadata[task];
            metadata.flags = SCHEDULER_TASK_EXECUTABLE | shape.extra_flags;
            metadata.active_mask = shape.mask;
            metadata.logical_block_num = shape.blocks;
            if (shape.blocks > 1) metadata.flags |= SCHEDULER_TASK_SPMD;
            if (shape.blocks > 1 && shape.sync_start) metadata.flags |= SCHEDULER_TASK_SYNC_START;
            if (shape.mask.count() > 1) metadata.flags |= SCHEDULER_TASK_MIX;
            metadata.trace_index_base = trace;
            trace += shape.blocks;
            metadata.timing_slot = -1;
            for (uint32_t slot = 0; slot < 3; ++slot)
                metadata.kernel_ids[slot] = shape.mask.has(static_cast<SubtaskSlot>(slot)) ? 1 : UINT16_MAX;
            if ((metadata.flags & SCHEDULER_TASK_INLINE) != 0) continue;
            if (shape.mask.count() > 1) graph.mixed(task, shape.mask);
            else graph.executable(task, first_lane(shape.mask));
        }
        storage.task_catalog_offset = 0;
        storage.calls_prepared = false;
        storage.prepare_catalog();
        ASSERT_TRUE(scheduler_init_spmd_task_records(base, storage.layout, epoch));
        for (uint64_t worker = 0; worker < worker_count; ++worker)
            storage.test_graphs[worker] = graph.graph();
        for (uint64_t worker = 0; worker < worker_count; ++worker) {
            storage.test_contexts[worker] = SchedulerLocalState{};
            storage.test_profiles[worker] = SchedulerLocalProfilingState{};
            locals[worker] = storage.local_context(&storage.contexts[worker]);
            locals[worker]->run_epoch = epoch;
        }
        entries.clear();
        kernel_run_count() = 0;
    }

    void *base() const { return storage.scheduler_state->base(); }
    SchedulerRunControl *run() const { return storage.run_control; }
    SchedulerTaskControl *control(uint64_t task) const {
        return scheduler_state_at<SchedulerTaskControl>(base(), storage.layout.task_controls_offset) + task;
    }
    void dependency(uint64_t consumer, std::vector<int32_t> producers) {
        graph.executable(consumer, first_lane(storage.metadata[consumer].active_mask), std::move(producers));
    }
    bool prepare() {
        storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&counting_kernel);
        return storage.prepare_calls(graph.graph());
    }
    bool refill(uint32_t worker, bool *progress = nullptr) {
        bool advanced = false;
        const bool ok = scheduler_refill_execution_tokens(graph.graph(), base(), locals[worker], run(), 0, &advanced);
        if (progress != nullptr) *progress = advanced;
        return ok;
    }
    bool fill(uint32_t worker) {
        bool failed = false;
        scheduler_find_local_execution(graph.graph(), base(), locals[worker], run(), 0, &failed);
        return !failed;
    }
    bool progress_tokens(uint32_t worker, uint32_t *selected) {
        bool progress = false;
        *selected = UINT32_MAX;
        return scheduler_progress_execution_tokens(
            graph.graph(), base(), locals[worker], run(), 0, selected, &progress
        );
    }
    // Claim and bind the named task on the worker, the way one discovery pass
    // of the resident loop would. False when the task is not yet issuable.
    bool prefetch(uint64_t task, uint32_t worker) {
        if (!prepare()) return false;
        auto *local = locals[worker];
        const auto &pool = *scheduler_state_at<SchedulerTaskCatalogHeader>(base(), local->config.task_catalog_offset);
        const auto catalog = scheduler_task_catalog(base(), pool, local->config.self_core_type);
        for (uint64_t index = 0; index < catalog.count; ++index) {
            if (catalog.refs[index] != static_cast<int64_t>(task)) continue;
            for (uint64_t visit = 0; visit <= catalog.count; ++visit) {
                bool progress = false;
                if (!scheduler_refill_execution_tokens(graph.graph(), base(), local, run(), 0, &progress)) return false;
                if (scheduler_execution_token_has_task(local, task)) return true;
                if (!progress) return false;
            }
            return false;
        }
        return false;
    }
    bool begin(uint32_t worker, Execution *execution) {
        auto *local = locals[worker];
        uint32_t slot = UINT32_MAX;
        if (!hbg_test_find_eligible(local, &slot)) return false;
        bool permit = false;
        if (!scheduler_check_entry_permit(graph.graph(), base(), local, run(), slot, &permit) || !permit) return false;
        if (!hbg_test_begin_execution(local, slot)) return false;
        const auto index = local->execution_at(slot).prepared.execution_index;
        auto *call = scheduler_prepared_call_at(base(), local->config.prepared, index);
        auto *binding = scheduler_execution_binding_at(base(), local->config.prepared, index);
        const uint64_t function =
            local->execution_at(slot).predicate == SchedulerPredicateState::FAIL ? 0 : call->function_bin_addr;
        *execution = {
            local->execution_at(slot).task_ref,
            static_cast<uint32_t>(binding->local_context.block_idx),
            static_cast<uint32_t>(binding->local_context.block_num),
            worker,
            slot,
            function
        };
        entries.emplace_back(execution->task, execution->block);
        if (function != 0)
            scheduler_execute_protocol_task(base(), local, slot, function, call->args, binding->local_context);
        return true;
    }
    bool complete(const Execution &execution) {
        return scheduler_complete_lane_execution<false>(
            graph.graph(), base(), locals[execution.worker], run(), execution.slot, 0
        );
    }
    uint32_t waiting_tokens(uint32_t worker) const {
        uint32_t count = 0;
        for (const auto &token : locals[worker]->execution_tokens)
            if (token.phase == SchedulerExecutionTokenPhase::WAIT_FANIN &&
                token.state != SchedulerDispatchSlotState::READY &&
                token.state != SchedulerDispatchSlotState::EXECUTING)
                ++count;
        return count;
    }
    uint32_t held_tokens(uint32_t worker) const { return scheduler_execution_token_count(locals[worker]); }
    bool done(uint64_t task) const {
        return scheduler_completion_done(scheduler_gm_fetch_add(control(task)->completion_state, UINT64_C(0)));
    }
    uint32_t completions(uint64_t task) const {
        return scheduler_completion_count(scheduler_gm_fetch_add(control(task)->completion_state, UINT64_C(0)));
    }
    uint64_t expected_entries() const {
        uint64_t count = 0;
        for (const auto &shape : shape_list) {
            const uint8_t flags = SCHEDULER_TASK_EXECUTABLE | shape.extra_flags |
                                  (shape.blocks > 1 ? SCHEDULER_TASK_SPMD : 0) |
                                  (shape.mask.count() > 1 ? SCHEDULER_TASK_MIX : 0);
            if (scheduler_task_uses_execution_tokens(flags)) count += shape.blocks * shape.mask.count();
        }
        return count;
    }
    // Drive every worker until the graph drains, with an explicit turn budget
    // so a stuck graph fails instead of hanging the test.
    bool drain(uint64_t turn_budget) {
        if (!prepare()) return false;
        uint64_t remaining = expected_entries();
        for (uint64_t turn = 0; turn < turn_budget && remaining != 0; ++turn) {
            for (uint32_t worker = 0; worker < worker_count; ++worker) {
                if (!fill(worker)) return false;
                Execution execution{};
                if (!begin(worker, &execution)) continue;
                if (!complete(execution)) return false;
                --remaining;
            }
        }
        return remaining == 0;
    }

    scheduler_test::FixtureStorage storage;
    scheduler_test::GraphBuffer graph;
    std::vector<Shape> shape_list;
    uint64_t worker_count{6};
    uint64_t run_epoch{0};
    std::array<SchedulerLocalState *, kMaxWorkers> locals{};
    std::vector<std::pair<int64_t, uint32_t>> entries;
};

}  // namespace scheduler_test::early_dispatch
