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

// Directory observations reserve neither a shared ready item nor physical members.
static inline __aicore__ bool scheduler_discover_cooperative_task(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool *progress
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarCooperativeDiscover);
    if (base == nullptr || local == nullptr || run == nullptr || progress == nullptr) return false;
    *progress = false;
    if (local->config.task_catalog_offset == 0)
        return scheduler_admission_fail(graph, local, run, -1, SchedulerErrorSite::ADMISSION_INVALID_CONFIG);
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    const auto &pool = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    const auto catalog = scheduler_task_catalog(base, pool, local->config.self_core_type);
    if (!scheduler_task_catalog_valid(catalog))
        return scheduler_admission_fail(graph, local, run, -1, SchedulerErrorSite::ADMISSION_INVALID_CONFIG);
    if (catalog.count == 0) return true;
    if (local->cooperative_scan_cursor >= catalog.count) local->cooperative_scan_cursor %= catalog.count;
    const int64_t task_ref = catalog.refs[local->cooperative_scan_cursor];
    if (++local->cooperative_scan_cursor == catalog.count) {
        local->cooperative_scan_cursor = 0;
        ++local->cooperative_scan_round;
    }
    if (!scheduler_task_ref_valid(graph, task_ref))
        return scheduler_admission_fail(graph, local, run, task_ref, SchedulerErrorSite::ADMISSION_INVALID_RECORD);
    const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
    if (!scheduler_task_is_executable(metadata.flags) || scheduler_task_is_inline(metadata.flags)) return true;
    if (!scheduler_task_requires_admission(metadata.flags)) return true;
    auto *admission = local->admission;
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
    *progress = staged == SchedulerAdmissionResult::PROGRESS;
    return true;
}
