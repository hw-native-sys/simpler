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

#include "scheduler_protocol_block.h"
#include "scheduler_protocol_execution.h"

enum class SchedulerAdmissionResult : uint8_t { PROGRESS, RETRY, ERROR };

// GENERAL worker IDs need not encode a role. Validate the complete frozen
// topology once, then use the run-entry-validated local lane and core type.
static inline __aicore__ bool
scheduler_validate_resident_topology(__gm__ void *base, const SchedulerLocalState *local) {
    SchedulerScalarScope scalar_scope(SchedPhaseKind::ScalarBootstrapTopology);
    const uint64_t workers = local->config.runtime_worker_count;
    const uint64_t cubes = local->config.cluster_count;
    if (workers == 0 || workers > SCHEDULER_WORKER_CAPACITY || cubes == 0 ||
        workers != cubes * PLATFORM_CORES_PER_BLOCKDIM)
        return false;
    uint64_t cube_words[(SCHEDULER_WORKER_CAPACITY + 63) / 64]{};
    uint64_t vector_words[2][(SCHEDULER_WORKER_CAPACITY + 63) / 64]{};
    uint64_t observed_cubes = 0;
    uint64_t observed_clusters = 0;
    static_assert(SCHEDULER_CLUSTER_CAPACITY <= 64);
    for (uint64_t worker = 0; worker < workers; ++worker) {
        auto *context = scheduler_worker_context_at(base, local, worker);
        scheduler_observe_cache_line(context);
        const int32_t type = context->core_type;
        if (context->worker_index != worker ||
            (type != static_cast<int32_t>(CoreType::AIC) && type != static_cast<int32_t>(CoreType::AIV)))
            return false;
        if (type == static_cast<int32_t>(CoreType::AIC)) {
            ++observed_cubes;
            cube_words[worker / 64] |= UINT64_C(1) << (worker % 64);
            scheduler_invalidate_cache_line(&context->cluster_index);
            scheduler_invalidate_cache_line(&context->cluster_worker_ids[0]);
            scheduler_invalidate_cache_line(&context->cluster_worker_ids[2]);
            scheduler_cache_barrier();
            if (context->cluster_index >= cubes || context->cluster_worker_ids[0] != worker ||
                (observed_clusters & (UINT64_C(1) << context->cluster_index)) != 0)
                return false;
            observed_clusters |= UINT64_C(1) << context->cluster_index;
            for (uint32_t lane = 1; lane < 3; ++lane) {
                const uint64_t peer = context->cluster_worker_ids[lane];
                if (peer >= workers || peer == worker) return false;
                auto *peer_context = scheduler_worker_context_at(base, local, peer);
                scheduler_invalidate_cache_line(peer_context);
                scheduler_invalidate_cache_line(&peer_context->cluster_index);
                scheduler_invalidate_cache_line(&peer_context->cluster_worker_ids[0]);
                scheduler_invalidate_cache_line(&peer_context->cluster_worker_ids[2]);
                scheduler_cache_barrier();
                if (peer_context->worker_index != peer ||
                    peer_context->core_type != static_cast<int32_t>(CoreType::AIV) ||
                    peer_context->cluster_index != context->cluster_index)
                    return false;
                for (uint32_t member_lane = 0; member_lane < PLATFORM_CORES_PER_BLOCKDIM; ++member_lane)
                    if (peer_context->cluster_worker_ids[member_lane] != context->cluster_worker_ids[member_lane])
                        return false;
                const uint64_t bit = UINT64_C(1) << (peer % 64);
                if (((vector_words[0][peer / 64] | vector_words[1][peer / 64]) & bit) != 0) return false;
                vector_words[lane - 1][peer / 64] |= bit;
            }
        }
    }
    if (observed_cubes != cubes) return false;
    const uint64_t words = (workers + 63) / 64;
    for (uint64_t word = 0; word < words; ++word) {
        const uint64_t remaining = workers - word * 64;
        const uint64_t valid = remaining >= 64 ? UINT64_MAX : (UINT64_C(1) << remaining) - 1;
        const uint64_t vectors = vector_words[0][word] | vector_words[1][word];
        if ((vectors & cube_words[word]) != 0 || (vectors | cube_words[word]) != valid) return false;
    }
    return true;
}

static inline __aicore__ bool scheduler_admission_fail(
    const SchedulerGraphAccess &graph, SchedulerLocalState *local, __gm__ SchedulerRunControl *run, int64_t task_ref,
    SchedulerErrorSite site, const SchedulerLocalSlotState *slot = nullptr
) {
    if (run != nullptr)
        scheduler_record_error(run, task_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local, site, slot);
    return false;
}

// The catalog was frozen after expanding all instances in this run. Physical
// topology is checked once, before any worker receives a group descriptor.
static inline __aicore__ bool scheduler_initialize_order(__gm__ void *base, SchedulerLocalState *local) {
    const auto &summary = *scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset);
    const auto &catalog = *scheduler_state_at<SchedulerTaskCatalogHeader>(base, local->config.task_catalog_offset);
    const uint64_t workers = local->config.runtime_worker_count;
    const uint32_t clusters = local->config.cluster_count;
    if (summary.worker_count != workers || summary.cluster_count != clusters || workers != 3 * clusters ||
        clusters == 0 || workers > SCHEDULER_WORKER_CAPACITY || catalog.task_count != local->config.graph->task_count ||
        summary.mix_task_count > catalog.task_count ||
        (summary.mix_task_count == 0) != (catalog.first_mix_ref == UINT32_MAX))
        return false;
    if (summary.topology_layout == SchedulerTopologyLayout::GENERAL) {
        if (!scheduler_validate_resident_topology(base, local)) return false;
    } else {
        if (summary.topology_layout != SchedulerTopologyLayout::COMPACT_LAUNCH) return false;
        const uint64_t worker = local->worker_id();
        const uint64_t cluster = worker < clusters ? worker : (worker - clusters) / 2;
        const uint32_t lane = worker < clusters ? 0U : 1U + static_cast<uint32_t>((worker - clusters) % 2);
        if (local->config.self_lane != lane || local->config.worker_ids[0] != cluster ||
            local->config.worker_ids[1] != clusters + 2 * cluster ||
            local->config.worker_ids[2] != clusters + 2 * cluster + 1)
            return false;
    }
    if (!scheduler_protocol_bind(base, local)) return false;
    local->run_epoch = summary.run_epoch;
    local->run_epoch_known = true;
    local->next_mix_ref = catalog.first_mix_ref;
    if (catalog.terminal_mix_refs[0] != UINT32_MAX) {
        const uint32_t block = scheduler_fixed_mix_block(local->config.cluster_index, clusters, LaneMask::all());
        for (unsigned stage = 0; stage < 2; ++stage) {
            const uint32_t ref = catalog.terminal_mix_refs[stage];
            if (ref >= catalog.task_count || !scheduler_acquire_group_lane(
                                                 base, local->config.prepared, local->worker_id(), ref, block,
                                                 local->config.self_lane, 7, &local->terminal_mix_calls[stage]
                                             ))
                return false;
            local->terminal_mix_refs[stage] = ref;
        }
    }
    return true;
}
