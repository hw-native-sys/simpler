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

#include "inner_kernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "test_hbg_owned_execution.h"
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_protocol_execute.h"
#include "scheduler/scheduler_args_dump.h"

namespace scheduler_test::gm_mix {

struct GmShape {
    uint16_t blocks;
    LaneMask mask;
    bool sync_start{false};
    bool inline_task{false};
};

// Every lane set that makes a task a Mix: the two AIC+single-AIV pairs, both
// AIVs, and all three.
constexpr LaneMask kAicAiv0{/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false};
constexpr LaneMask kAicAiv1{/*aic=*/true, /*aiv0=*/false, /*aiv1=*/true};
constexpr std::array<LaneMask, 4> kMixMasks{kAicAiv0, kAicAiv1, LaneMask::both_aiv(), LaneMask::all()};

// gtest cannot stream a LaneMask, so SCOPED_TRACE names the lanes instead.
inline std::string lane_names(LaneMask lanes) {
    std::string name;
    if (lanes.has_aic()) name += "aic";
    if (lanes.has_aiv0()) name += name.empty() ? "aiv0" : "+aiv0";
    if (lanes.has_aiv1()) name += name.empty() ? "aiv1" : "+aiv1";
    return name.empty() ? "none" : name;
}

inline uint64_t gm_trace_capacity(const std::vector<GmShape> &shapes) {
    uint64_t maximum = 1;
    for (const auto &shape : shapes)
        maximum = std::max<uint64_t>(maximum, static_cast<uint64_t>(shape.blocks) * 3);
    return maximum;
}

inline SchedulerSpmdRequirements gm_requirements(const std::vector<GmShape> &shapes, uint32_t clusters) {
    SchedulerSpmdRequirements result{clusters * 3, clusters, 0, 0};
    for (const auto &shape : shapes) {
        const bool mix = shape.mask.is_mix();
        if (shape.sync_start) ++result.cooperative_task_count;
        if (mix) ++result.mix_task_count;
        if (mix && shape.sync_start) ++result.sync_mix_task_count;
    }
    return result;
}

struct GmExecution {
    int64_t task{SCHEDULER_TASK_REF_INVALID};
    uint32_t block{0};
    uint32_t blocks{0};
    uint32_t worker{0};
    uint32_t slot{0};
    uint8_t subtask{UINT8_MAX};
    uint32_t physical_subblock{0};
};

struct GmMixFixture {
    explicit GmMixFixture(
        const std::vector<GmShape> &shapes, bool protocol_profiling = false, uint32_t clusters = 2,
        const std::function<void(AicoreSchedulerLayout &)> &extend_layout = {}
    ) :
        storage(
            shapes.size(), clusters * 3, gm_trace_capacity(shapes), gm_requirements(shapes, clusters),
            protocol_profiling, extend_layout
        ),
        graph(shapes.size()),
        locals(clusters * 3),
        signals(clusters) {
        uint32_t trace = 0;
        for (uint64_t task = 0; task < shapes.size(); ++task) {
            const auto shape = shapes[task];
            graph.mixed(task, shape.mask);
            auto &metadata = storage.metadata[task];
            metadata.active_mask = shape.mask;
            metadata.logical_block_num = shape.blocks;
            metadata.flags = SCHEDULER_TASK_EXECUTABLE;
            if (shape.blocks > 1) metadata.flags |= SCHEDULER_TASK_SPMD;
            if (shape.sync_start) metadata.flags |= SCHEDULER_TASK_SYNC_START;
            if (shape.mask.is_mix()) metadata.flags |= SCHEDULER_TASK_MIX;
            if (shape.inline_task) {
                EXPECT_EQ(shape.blocks, 1u);
                EXPECT_FALSE(shape.sync_start);
                EXPECT_EQ(shape.mask.count(), 1u);
                metadata.flags |= SCHEDULER_TASK_INLINE;
                for (auto &kernel : graph.storage()[task].task.kernel_id)
                    kernel = INVALID_KERNEL_ID;
            }
            metadata.trace_index_base = trace;
            trace += shape.blocks * static_cast<uint32_t>(shape.mask.count());
            metadata.timing_slot = -1;
            for (uint8_t subtask = 0; subtask < 3; ++subtask)
                metadata.kernel_ids[subtask] =
                    !shape.inline_task && shape.mask.has(static_cast<SubtaskSlot>(subtask)) ? 1 : UINT16_MAX;
        }
        storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&scheduler_test::protocol_noop_kernel);
        storage.prepare_catalog();
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, 47));
        storage.run_control->active_worker_count = clusters * 3;

        for (uint32_t worker = 0; worker < locals.size(); ++worker) {
            auto &context = storage.contexts[worker];
            for (uint32_t lane = 0; lane < 3; ++lane)
                context.cluster_worker_ids[lane] = (worker / 3) * 3 + lane;
            storage.test_graphs[worker] = graph.graph();
        }
        for (auto &signal : signals)
            scheduler_cluster_signal_clear(&signal, 47);
        for (uint32_t worker = 0; worker < locals.size(); ++worker) {
            locals[worker] = storage.local_context(&storage.contexts[worker]);
            if (storage.layout.spmd.mix_task_count != 0) locals[worker]->cluster_signal = &signals[worker / 3];
            const auto &catalog = *scheduler_state_at<SchedulerTaskCatalogHeader>(base(), storage.task_catalog_offset);
            locals[worker]->next_mix_ref = catalog.first_mix_ref;
            if (protocol_profiling) EXPECT_TRUE(scheduler_protocol_bind(base(), locals[worker]));
        }
    }

    void *base() const { return storage.scheduler_state->base(); }
    SchedulerRunControl *run() const { return storage.run_control; }
    SchedulerTaskControl *control(uint64_t task) const {
        return scheduler_state_at<SchedulerTaskControl>(base(), storage.layout.task_controls_offset) + task;
    }
    SchedulerClusterSignal *signal(uint32_t cluster) { return &signals[cluster]; }
    bool ordinary_arm(uint64_t task, uint32_t cluster, bool *published) {
        *published = false;
        auto *proxy = locals[cluster * 3 + 1];
        const auto task_ref = static_cast<int64_t>(task);
        bool progress = false;
        if (scheduler_task_ref_is_null(proxy->cluster_pending.task_ref)) {
            int64_t ref = SCHEDULER_TASK_REF_INVALID;
            if (!scheduler_issue_head(graph.graph(), base(), proxy, run(), &ref, &progress)) return false;
            if (ref != task_ref) return true;
            if (!scheduler_claim_for_build(graph.graph(), base(), proxy, run(), ref, &progress)) return false;
        }
        uint32_t previous = 0;
        for (const auto &group : signals[cluster].groups)
            previous = std::max(previous, static_cast<uint32_t>(group.armed_generation));
        if (!scheduler_progress_mix_publication(graph.graph(), base(), proxy, run(), &progress)) return false;
        for (const auto &group : signals[cluster].groups) {
            const uint32_t generation = group.armed_generation;
            if (scheduler_cluster_mix_armed(generation) && generation > previous && group.task_ref == task_ref)
                *published = true;
        }
        return true;
    }
    bool ordinary_service(uint32_t worker, bool *progress, uint64_t profiling = 0) {
        *progress = false;
        return scheduler_progress_mix_groups(graph.graph(), base(), locals[worker], run(), profiling, progress);
    }
    bool prepare_calls() { return storage.prepare_calls(graph.graph()); }
    bool claim_ordinary(uint64_t task, uint32_t worker) {
        if (!prepare_calls()) return false;
        auto *local = locals[worker];
        const auto *pool = scheduler_state_at<SchedulerTaskCatalogHeader>(base(), local->config.task_catalog_offset);
        const auto catalog = scheduler_task_catalog(base(), *pool, local->config.self_core_type);
        const auto *entry = std::find(catalog.refs, catalog.refs + catalog.count, static_cast<int64_t>(task));
        if (entry == catalog.refs + catalog.count) return false;
        for (uint64_t slice = 0; slice <= catalog.count + 2 * graph.graph().task_count; ++slice) {
            const uint64_t head = local->issue_cursor;
            const int64_t issue_task = local->issue_task_ref;
            const uint32_t fanin = local->issue_next_fanin;
            const uint32_t tokens = scheduler_execution_token_count(local);
            bool progress = false;
            if (!scheduler_refill_execution_tokens(graph.graph(), base(), local, run(), 0, &progress)) return false;
            if (std::any_of(
                    std::begin(local->execution_tokens), std::end(local->execution_tokens), [&](const auto &token) {
                        return token.task_ref == static_cast<int64_t>(task) &&
                               token.phase == SchedulerExecutionTokenPhase::WAIT_FANIN &&
                               token.prepared.claimed_state != 0;
                    }
                ))
                return true;
            if (scheduler_execution_token_count(local) != tokens) return false;
            if (local->issue_cursor == head && !(local->issue_next_fanin != 0 && (local->issue_task_ref != issue_task ||
                                                                                  local->issue_next_fanin > fanin)))
                return false;
        }
        return false;
    }
    bool progress_ordinary(uint32_t worker) {
        auto *local = locals[worker];
        uint32_t direct = UINT32_MAX;
        bool progress = false;
        return scheduler_progress_execution_tokens(graph.graph(), base(), local, run(), 0, &direct, &progress);
    }
    bool task_done(uint64_t task) const {
        return scheduler_completion_done(scheduler_gm_query(control(task)->completion_state));
    }
    template <bool RecordProtocol = true>
    bool fill(uint32_t worker) {
        bool failed = false;
        scheduler_find_local_execution<RecordProtocol>(graph.graph(), base(), locals[worker], run(), 0, &failed);
        return !failed;
    }
    bool service_cooperative(uint32_t worker) {
        bool progress = false;
        return ordinary_service(worker, &progress);
    }
    bool begin(uint32_t worker, GmExecution *execution) {
        uint32_t slot = UINT32_MAX;
        return hbg_test_find_eligible(locals[worker], &slot) && begin(worker, slot, execution);
    }
    bool begin(uint32_t worker, uint32_t slot, GmExecution *execution) {
        auto *local = locals[worker];
        if (slot >= SCHEDULER_LOCAL_EXECUTION_COUNT ||
            local->execution_at(slot).state != SchedulerDispatchSlotState::READY)
            return false;
        bool permit = false;
        if (!scheduler_check_entry_permit(graph.graph(), base(), local, run(), slot, &permit)) return false;
        if (!permit) return false;
        if (!hbg_test_begin_execution(local, slot)) return false;
        const auto index = local->execution_at(slot).prepared.execution_index;
        auto *call = scheduler_prepared_call_at(base(), local->config.prepared, index);
        auto *binding = scheduler_execution_binding_at(base(), local->config.prepared, index);
        const auto address =
            local->execution_at(slot).predicate == SchedulerPredicateState::FAIL ? 0 : call->function_bin_addr;
        const GmExecution entry = {
            local->execution_at(slot).task_ref,
            static_cast<uint32_t>(binding->local_context.block_idx),
            static_cast<uint32_t>(binding->local_context.block_num),
            worker,
            slot,
            local->execution_at(slot).subtask_slot,
            static_cast<uint32_t>(binding->global_context.sub_block_id)
        };
        scheduler_execute_protocol_task(base(), local, slot, address, call->args, binding->local_context);
        *execution = entry;
        return true;
    }
    template <bool RecordProtocol = true>
    bool complete(const GmExecution &execution, uint64_t profiling = 0) {
        auto *local = locals[execution.worker];
        return scheduler_complete_lane_execution<RecordProtocol>(
            graph.graph(), base(), local, run(), execution.slot, profiling
        );
    }
    bool drain(uint32_t worker, bool *done) {
        return scheduler_drain_ordered_exit(graph.graph(), base(), locals[worker], run(), done);
    }

    scheduler_test::FixtureStorage storage;
    scheduler_test::GraphBuffer graph;
    std::vector<SchedulerLocalState *> locals;
    std::vector<SchedulerClusterSignal> signals;
};

}  // namespace scheduler_test::gm_mix
