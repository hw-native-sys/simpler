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

#include "ready_pool_plan.h"
#include "scheduler/scheduler_admission_members.h"

// The host owns this image until the launch upload. Only the ordinary CSR
// path has all descriptors and initial dependency credits available here.
inline bool scheduler_seed_ordinary_bootstrap(
    void *base, const AicoreSchedulerLayout &layout, const SchedulerReadyPoolPlan &plan, uint64_t epoch
) {
    const auto &pool = plan.header;
    if (pool.dependencies_offset == 0) return true;
    const uint64_t clusters = layout.spmd.cluster_count;
    const uint64_t workers = layout.spmd.worker_count;
    if (clusters == 0 || workers != 3 * clusters || workers > SCHEDULER_WORKER_CAPACITY ||
        pool.task_count != layout.task_count || plan.descriptors.size() != SCHEDULER_READY_QUEUE_COUNT)
        return false;

    constexpr uint64_t max_words = (SCHEDULER_WORKER_CAPACITY + 63) / 64;
    uint64_t cube_words[max_words]{};
    uint64_t vector0_words[max_words]{};
    scheduler_compact_member_masks(clusters, cube_words, vector0_words);
    const auto cluster_of = [clusters](uint64_t worker) {
        return worker < clusters ? worker : (worker - clusters) / 2;
    };
    const auto *dependencies = scheduler_state_at<SchedulerDependencyHeader>(base, pool.dependencies_offset);
    const auto *remaining = scheduler_state_at<uint64_t>(base, dependencies->remaining_offset);
    const auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(base, layout.task_metadata_offset);
    auto *controls = scheduler_state_at<SchedulerTaskControl>(base, layout.task_controls_offset);
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        const auto &meta = metadata[task];
        // Executable inline nodes still need a ticket to publish TaskDone.
        if (!scheduler_task_is_executable(meta.flags)) continue;
        if (remaining[task] > task) return false;
        if (scheduler_task_requires_admission(meta)) {
            auto *record = scheduler_state_at<SchedulerAdmissionRecord>(
                base, layout.admission_records_offset + task * layout.admission_record_stride
            );
            // Keep even-sized AIV waves on whole clusters so an independent
            // MIX wave can use the complementary clusters without fragmentation.
            if (layout.spmd.mix_task_count != 0 && !scheduler_task_is_mix(meta.flags) && !meta.active_mask.has_aic() &&
                !scheduler_task_requires_sync_start(meta.flags) && meta.logical_block_num < workers - clusters &&
                meta.logical_block_num % 2 == 0) {
                record->member_rank_offset += record->member_rank_offset % 2;
            }
            auto *words = scheduler_state_at<uint64_t>(base, record->member_words_offset);
            if ((meta.flags & UINT8_C(128)) != 0 || scheduler_task_has_predicate(meta.flags) ||
                !scheduler_select_admission_members(
                    *record, workers, clusters, cube_words, vector0_words, cluster_of, words, &record->member_count
                ))
                return false;
            record->run_epoch = epoch;
            controls[task].admission_state = static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED);
        }
        if (remaining[task] == 0) {
            const auto queue =
                scheduler_ready_pool_queue(base, pool, scheduler_task_ready_queue(meta.flags, meta.active_mask));
            if (!scheduler_ticket_view_valid(queue) || queue.control->reserved_tail >= queue.capacity) return false;
            const uint64_t position = queue.control->reserved_tail;
            queue.entries[position].task_ref = static_cast<int64_t>(task);
            queue.control->reserved_tail = position + 1;
        }
    }
    auto *run = scheduler_state_at<SchedulerRunControl>(base, layout.run_control_offset);
    run->bootstrap_scan_complete = SCHEDULER_BOOTSTRAP_HOST_SEEDED;
    return true;
}
