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

#include "scheduler_cluster_mix.h"

// The shared issuance head names the only cohort eligible for new admission.
static inline __aicore__ bool scheduler_discover_cooperative_task(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress, SchedulerIssueSnapshot *snapshot = nullptr
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarCooperativeDiscover);
    if (base == nullptr || local == nullptr || run == nullptr || progress == nullptr) return false;
    *progress = false;
    if (local->config.task_catalog_offset == 0)
        return scheduler_admission_fail(graph, local, run, -1, SchedulerErrorSite::ADMISSION_INVALID_CONFIG);
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    auto *admission = local->admission;
    if (admission != nullptr && admission->initialized) {
        if (admission->scan_cursor > admission->capacity ||
            (admission->scan_cursor == admission->capacity && !scheduler_task_ref_is_null(admission->current_task_ref)))
            return scheduler_admission_fail(
                graph, local, run, admission->current_task_ref, SchedulerErrorSite::ADMISSION_INVALID_FRONTIER
            );
        // The cursor advances only after consuming or closing each finite log entry.
        if (admission->scan_cursor == admission->capacity || !scheduler_task_ref_is_null(admission->pending_task_ref))
            return true;
    }
    const auto &pool = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    const auto catalog = scheduler_task_catalog(base, pool, local->config.self_core_type);
    if (!scheduler_task_catalog_valid(catalog))
        return scheduler_admission_fail(graph, local, run, -1, SchedulerErrorSite::ADMISSION_INVALID_CONFIG);
    if (catalog.count == 0) return true;
    int64_t task_ref = SCHEDULER_TASK_REF_INVALID;
    if (!scheduler_issue_head(graph, base, local, run, &task_ref, progress, snapshot)) return false;
    if (scheduler_task_ref_is_null(task_ref)) return true;
    if (!scheduler_task_ref_valid(graph, task_ref))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
    const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
    if (!scheduler_task_is_executable(metadata.flags) || scheduler_task_is_inline(metadata.flags)) return true;
    if (!scheduler_task_requires_admission(metadata.flags)) return true;
    if (admission == nullptr || admission->capacity == 0)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::ADMISSION_INVALID_CONFIG);
    if (!scheduler_task_ref_is_null(admission->pending_task_ref)) return true;
    if (scheduler_gm_query(scheduler_task_control_at(base, local, task_ref)->admission_state) ==
        static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED))
        return true;
    SchedulerFaninView fanin{};
    if (scheduler_node_fanin_view(graph, task_ref, &fanin) != SchedulerGraphResult::OK)
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
    uint32_t next_fanin = 0;
    bool ready = false;
    if (!scheduler_observe_fanin_prefix(graph, base, local, run, task_ref, fanin, next_fanin, &ready)) return false;
    if (!ready) return true;
    const auto staged = scheduler_stage_admission_candidate(graph, base, local, run, admission, task_ref);
    if (staged == SchedulerAdmissionResult::ERROR) return false;
    *progress = *progress || staged == SchedulerAdmissionResult::PROGRESS;
    return true;
}
