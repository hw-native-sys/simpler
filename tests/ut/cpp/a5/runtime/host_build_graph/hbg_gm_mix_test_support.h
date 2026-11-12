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
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_gm_mix.h"
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_protocol_execute.h"
#include "scheduler/scheduler_args_dump.h"

namespace scheduler_test::gm_mix {

struct GmShape {
    uint16_t blocks;
    LaneMask mask;
    bool sync_start{false};
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
        if (mix || shape.blocks > 1) ++result.cooperative_task_count;
        if (mix) ++result.mix_task_count;
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
    explicit GmMixFixture(const std::vector<GmShape> &shapes, bool protocol_profiling = false, uint32_t clusters = 2) :
        storage(
            shapes.size(), clusters * 3, gm_trace_capacity(shapes), gm_requirements(shapes, clusters),
            protocol_profiling
        ),
        graph(shapes.size()),
        admissions(clusters * 3),
        locals(clusters * 3) {
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
            metadata.trace_index_base = trace;
            trace += shape.blocks * static_cast<uint32_t>(shape.mask.count());
            metadata.timing_slot = -1;
            for (uint8_t subtask = 0; subtask < 3; ++subtask)
                metadata.kernel_ids[subtask] = shape.mask.has(static_cast<SubtaskSlot>(subtask)) ? 1 : UINT16_MAX;
        }
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, 47));
        storage.prepare_ready_pool();
        storage.run_control->active_worker_count = clusters * 3;
        storage.run_control->aic_active_worker_count = clusters;
        storage.run_control->aiv_active_worker_count = clusters * 2;
        storage.run_control->expected_task_count = shapes.size();
        for (uint32_t worker = 0; worker < locals.size(); ++worker) {
            auto &context = storage.contexts[worker];
            context.is_scheduler = 1;
            context.scheduler_count = clusters * 3;
            for (uint32_t lane = 0; lane < 3; ++lane)
                context.cluster_worker_ids[lane] = (worker / 3) * 3 + lane;
            storage.test_graphs[worker] = graph.graph();
        }
        for (uint32_t worker = 0; worker < locals.size(); ++worker) {
            locals[worker] = storage.local_context(&storage.contexts[worker]);
            EXPECT_TRUE(scheduler_initialize_admission(base(), locals[worker], &admissions[worker]));
            locals[worker]->admission = &admissions[worker];
        }
    }

    void *base() const { return storage.scheduler_state->base(); }
    SchedulerRunControl *run() const { return storage.run_control; }
    SchedulerTaskControl *control(uint64_t task) const {
        return scheduler_state_at<SchedulerTaskControl>(base(), storage.layout.task_controls_offset) + task;
    }
    SchedulerGroupMailbox *mailbox(uint32_t cluster) const {
        return scheduler_state_at<SchedulerGroupMailbox>(base(), storage.layout.group_mailboxes_offset) + cluster;
    }
    volatile uint64_t &seal(uint64_t task, uint32_t cluster) const {
        const auto *record = scheduler_state_at<SchedulerAdmissionRecord>(
            base(), storage.layout.admission_records_offset + task * storage.layout.admission_record_stride
        );
        return scheduler_state_at<SchedulerMixCloseSeal>(base(), record->mix_close_seals_offset)[cluster].value;
    }
    bool enqueue(uint64_t task, uint32_t worker = 0) {
        auto *local = locals[worker];
        if (scheduler_route_task(graph.graph(), base(), local, run(), task) != SchedulerRouteResult::READY_TO_ENQUEUE)
            return false;
        SchedulerReadyBatch batch{};
        const auto &metadata = storage.metadata[task];
        return scheduler_ready_batch_append(base(), local, task, &batch) &&
               scheduler_ready_batch_push(
                   base(), local, scheduler_task_ready_queue(metadata.flags, metadata.active_mask), &batch
               );
    }
    bool claim_pending(uint64_t task, uint32_t worker = 0) {
        if (!enqueue(task, worker)) return false;
        auto *local = locals[worker];
        const auto &metadata = storage.metadata[task];
        const uint32_t queue = scheduler_task_ready_queue(metadata.flags, metadata.active_mask);
        SchedulerReadyClaim ready{};
        return scheduler_claim_ready_for_slot(graph.graph(), base(), local, run(), locals.size(), queue, &ready) &&
               ready.task_ref == static_cast<int64_t>(task) &&
               scheduler_try_claim_admission(graph.graph(), base(), local, run(), &admissions[worker], ready) ==
                   SchedulerAdmissionResult::PROGRESS;
    }
    bool admit(uint64_t task, uint32_t worker = 0) {
        return claim_pending(task, worker) &&
               scheduler_prepare_admission_record(graph.graph(), base(), locals[worker], run(), &admissions[worker]) &&
               scheduler_try_commit_admission(graph.graph(), base(), locals[worker], run(), &admissions[worker]) ==
                   SchedulerAdmissionResult::PROGRESS;
    }
    bool refresh(uint32_t worker) {
        return scheduler_refresh_local_obligation(graph.graph(), base(), locals[worker], run(), &admissions[worker]);
    }
    bool refresh_all() {
        for (uint32_t worker = 0; worker < locals.size(); ++worker)
            if (!refresh(worker)) return false;
        return true;
    }
    bool participant(uint32_t worker, uint32_t task = 0) const {
        return storage.metadata[task]
            .active_mask.merged_with(LaneMask::aic())
            .has(static_cast<SubtaskSlot>(worker % 3));
    }
    SchedulerBlockClaimResult claim(uint32_t cluster) {
        return scheduler_gm_mix_claim(graph.graph(), base(), locals[cluster * 3], run());
    }
    bool publish(uint32_t cluster) {
        return scheduler_gm_mix_publish(graph.graph(), base(), locals[cluster * 3], run());
    }
    bool observe(uint32_t worker, bool *progress) {
        return scheduler_gm_mix_observe(graph.graph(), base(), locals[worker], run(), progress);
    }
    bool arrive(uint32_t worker, bool *progress) {
        return scheduler_gm_mix_arrive(graph.graph(), base(), locals[worker], run(), progress);
    }
    bool prepare(uint32_t worker, bool *progress, uint64_t profiling = 0) {
        return scheduler_gm_mix_prepare_lane(graph.graph(), base(), locals[worker], run(), profiling, progress);
    }
    bool rendezvous(uint32_t cluster, uint64_t profiling = 0) {
        bool progress = false;
        const LaneMask resources = mailbox(cluster)->descriptor.resource_mask;
        for (uint32_t lane = 1; lane < 3; ++lane)
            if (resources.has(static_cast<SubtaskSlot>(lane)) && !observe(cluster * 3 + lane, &progress)) return false;
        for (uint32_t lane = 0; lane < 3; ++lane)
            if (resources.has(static_cast<SubtaskSlot>(lane)) && !arrive(cluster * 3 + lane, &progress)) return false;
        for (uint32_t lane = 0; lane < 3; ++lane)
            if (resources.has(static_cast<SubtaskSlot>(lane)) && !prepare(cluster * 3 + lane, &progress, profiling))
                return false;
        return true;
    }
    SchedulerGmMixResult retire(uint32_t cluster) {
        return scheduler_gm_mix_retire(graph.graph(), base(), locals[cluster * 3], run());
    }
    bool complete_task(uint64_t task, uint32_t cluster) {
        return scheduler_complete_ordered_task(graph.graph(), base(), locals[cluster * 3], run(), task, 0);
    }
    bool close(uint32_t worker, bool *closed) {
        return scheduler_gm_mix_close(graph.graph(), base(), locals[worker], run(), closed);
    }
    template <bool RecordProtocol = true>
    bool fill(uint32_t worker) {
        bool failed = false;
        scheduler_fill_own_slots<RecordProtocol>(graph.graph(), base(), locals[worker], run(), 0, &failed);
        return !failed;
    }
    bool begin(uint32_t worker, GmExecution *execution) {
        auto *local = locals[worker];
        uint32_t slot = UINT32_MAX;
        if (!scheduler_local_ready_peek(local, &slot)) return false;
        bool permit = false;
        if (!scheduler_ordered_can_enter_slot(graph.graph(), base(), local, run(), slot, &permit)) return false;
        if (!permit) return false;
        if (!scheduler_local_ready_execute(local, slot)) return false;
        auto *payload =
            scheduler_state_at<DispatchPayload>(base(), local->dispatch_payload_offset(local->config.self_lane, slot));
        if (scheduler_protocol_enabled(local)) scheduler_execute_protocol_task(base(), local, slot, payload);
        *execution = {
            local->slots[slot].task_ref,
            static_cast<uint32_t>(payload->local_context.block_idx),
            static_cast<uint32_t>(payload->local_context.block_num),
            worker,
            slot,
            local->slots[slot].subtask_slot,
            static_cast<uint32_t>(payload->global_context.sub_block_id)
        };
        return true;
    }
    template <bool RecordProtocol = true>
    bool complete(const GmExecution &execution) {
        auto *local = locals[execution.worker];
        return scheduler_complete_own_slot<RecordProtocol>(graph.graph(), base(), local, run(), execution.slot, 0);
    }
    bool finish_lanes(uint32_t cluster) {
        const auto descriptor = mailbox(cluster)->descriptor;
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (!descriptor.resource_mask.has(static_cast<SubtaskSlot>(lane))) continue;
            GmExecution execution{};
            const bool active = descriptor.lane_subtasks[lane] != UINT8_MAX;
            const bool entered = begin(cluster * 3 + lane, &execution);
            if (active != entered || (entered && !complete(execution))) return false;
        }
        return true;
    }
    bool drain(uint32_t worker, bool *done) {
        return scheduler_drain_ordered_exit(graph.graph(), base(), locals[worker], run(), done);
    }

    scheduler_test::FixtureStorage storage;
    scheduler_test::GraphBuffer graph;
    std::vector<SchedulerAdmissionLocalState> admissions;
    std::vector<SchedulerLocalState *> locals;
};

}  // namespace scheduler_test::gm_mix
