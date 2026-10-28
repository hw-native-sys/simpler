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

#include "scheduler_graph.h"
#include "scheduler_memory.h"
#include "scheduler_types.h"

struct SchedulerErrorIdentity {
    uint64_t run_epoch{UINT64_MAX};
    uint64_t order_key{UINT64_MAX};
    uint64_t block_idx{UINT64_MAX};
    uint64_t group_seq{UINT64_MAX};
    uint64_t worker_index{UINT64_MAX};
    uint32_t cluster_lane{UINT32_MAX};
    uint32_t valid_fields{0};
};

static_assert(sizeof(SchedulerErrorIdentity) == 48, "error identity capture size changed");

// The status line has a single writer and never uses AICore atomics. A
// device load therefore reads the publication itself, not an atomic's
// potentially undrained copy. Identity fields retain their atomic access.
inline __aicore__ uint64_t scheduler_read_error(__gm__ SchedulerRunControl *run_control) {
    return scheduler_gm_query(run_control->scheduler_error);
}

inline __aicore__ void scheduler_publish_error(__gm__ SchedulerRunControl *run_control, uint64_t status) {
    scheduler_publish_signal(run_control->scheduler_error, status);
}

inline __aicore__ void scheduler_record_error(
    __gm__ SchedulerRunControl *run_control, int64_t task_ref, SchedulerGraphResult status,
    const SchedulerGraphView *graph = nullptr, __gm__ const SchedulerWorkerContext *context = nullptr,
    SchedulerErrorSite error_site = SchedulerErrorSite::UNKNOWN, const SchedulerErrorIdentity *identity = nullptr
) {
    if (run_control == nullptr || status == SchedulerGraphResult::OK) return;
    if (scheduler_gm_compare_exchange(run_control->error_claimed, UINT64_C(0), UINT64_C(1)) != 0) return;
    scheduler_gm_store(run_control->error_task_ref, static_cast<uint64_t>(task_ref));
    if (graph != nullptr) {
        scheduler_gm_store(run_control->error_graph_task_count, graph->task_count);
        scheduler_gm_store(run_control->error_storage_address, graph->storage_address);
        scheduler_gm_store(run_control->error_task_window_last_index, graph->task_window_last_index);
    }
    scheduler_gm_store(
        run_control->error_core_id, context != nullptr ? static_cast<uint64_t>(context->physical_core_id) : UINT64_MAX
    );
    scheduler_gm_store(
        run_control->error_core_type, context != nullptr ? static_cast<uint64_t>(context->core_type) : UINT64_MAX
    );
    SchedulerErrorIdentity fallback{};
    if (context != nullptr && context->worker_index < context->runtime_worker_count) {
        fallback.worker_index = context->worker_index;
        fallback.valid_fields |= SCHEDULER_ERROR_HAS_WORKER;
        uint32_t matches = 0;
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            if (context->cluster_worker_ids[lane] != context->worker_index) continue;
            fallback.cluster_lane = lane;
            ++matches;
        }
        if (matches == 1) fallback.valid_fields |= SCHEDULER_ERROR_HAS_LANE;
        else fallback.cluster_lane = UINT32_MAX;
    }
    const auto &captured = identity != nullptr ? *identity : fallback;
    scheduler_gm_store(run_control->error_run_epoch, captured.run_epoch);
    scheduler_gm_store(run_control->error_order_key, captured.order_key);
    scheduler_gm_store(run_control->error_block_idx, captured.block_idx);
    scheduler_gm_store(run_control->error_group_seq, captured.group_seq);
    scheduler_gm_store(run_control->error_worker_index, captured.worker_index);
    scheduler_gm_store(run_control->error_cluster_lane, captured.cluster_lane);
    scheduler_gm_store(run_control->error_identity_fields, captured.valid_fields);
    scheduler_gm_store(run_control->error_site, static_cast<uint64_t>(error_site));
    scheduler_cache_barrier();
    scheduler_publish_error(run_control, static_cast<uint64_t>(status));
}
