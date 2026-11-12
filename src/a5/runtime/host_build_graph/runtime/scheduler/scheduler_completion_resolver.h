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

#include "scheduler_ready_publication.h"

// CCEC has failed when over-aligned stack objects escape through an outlined
// call (#2973). Local arguments at this boundary use natural alignment; shared
// scheduler records are passed through their explicitly qualified GM pointers.
static_assert(
    alignof(SchedulerGraphAccess) <= 8 && alignof(SchedulerLocalState) <= 8 && alignof(SchedulerReadyClaim) <= 8 &&
        alignof(SchedulerReadyPublication) <= 8,
    "outlined resolver local arguments must not require over-aligned stack storage"
);

// Keep queue reservation/publication and retry state out of the synchronous
// resolver's direct-handoff path. Only a nonempty batch crosses this boundary.
static __aicore__ __attribute__((noinline)) bool scheduler_flush_nonempty_ready_publication(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t producer_ref, SchedulerReadyPublication &publication
) {
    uint32_t retries = 0;
    for (;;) {
        const auto result = scheduler_ready_publication_step(graph, base, local, publication);
        if (result == SchedulerReadyPublishResult::ERROR) {
            scheduler_record_error(
                run, producer_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                SchedulerErrorSite::COMPLETION_READY_PUBLISH_FAILED
            );
            return false;
        }
        if (result == SchedulerReadyPublishResult::DONE) break;
        if (++retries % 32 == 0 && scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    }
    // Only refs[0, count) are live. The first append initializes routing.
    publication.count = 0;
    publication.phase = SchedulerReadyPublishPhase::COLLECTING;
    return true;
}

static inline __aicore__ bool scheduler_flush_ready_publication(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t producer_ref, SchedulerReadyPublication &publication
) {
    return publication.count == 0 ||
           scheduler_flush_nonempty_ready_publication(graph, base, local, run, producer_ref, publication);
}

// Append validates metadata on its successful path. Error classification reads
// it only after append fails, preserving the specific run diagnostic.
static __aicore__ __attribute__((noinline)) bool scheduler_fail_ready_append(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t producer_ref, int64_t waiter_ref
) {
    auto error = SchedulerGraphResult::INVALID_ARGUMENTS;
    auto site = SchedulerErrorSite::COMPLETION_READY_APPEND_FAILED;
    SchedulerNodeKind kind{};
    const bool native = graph.native_header_offset != 0;
    if (native && scheduler_node_kind(graph, waiter_ref, &kind) != SchedulerGraphResult::OK) {
        site = SchedulerErrorSite::COMPLETION_INVALID_WAITER;
    } else if (!native || !scheduler_node_is_control(kind)) {
        const auto &metadata = *scheduler_task_metadata_at(base, local, waiter_ref);
        if (!scheduler_task_is_executable(metadata.flags)) {
            site = SchedulerErrorSite::COMPLETION_WAITER_NOT_EXECUTABLE;
        } else if (scheduler_metadata_single_subtask_slot(scheduler_task_lanes(metadata)) == UINT8_MAX &&
                   !scheduler_task_is_mix(metadata.flags)) {
            error = SchedulerGraphResult::UNSUPPORTED_SHAPE;
            site = SchedulerErrorSite::COMPLETION_INVALID_SHAPE;
        }
    }
    scheduler_record_error(run, producer_ref, error, &graph, local, site);
    return false;
}

// The caller is the unique TaskDone owner. CSR storage is Host-validated and
// immutable until the run fence; it contains no dynamically registered waiter.
// scheduler_init_dependencies validates/deduplicates input edges and constructs
// bounded, ordered CSR rows; run entry validates the storage layout. Therefore
// producer_ref and every row reference already index this graph's tables.
// This overload finishes synchronously and never exports a continuation.
// Contribution or inline completion checks cancellation before TaskDone. This
// same-call continuation shares that observation; long fanout refreshes it.
static __aicore__ __attribute__((noinline)) bool scheduler_resolve_csr_completion(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t producer_ref, const __gm__ SchedulerDependencyHeader &dependencies, uint64_t profiling_level,
    SchedulerReadyClaim *direct_ready, uint32_t direct_core_type, bool admission_owner
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarResolve, producer_ref);
    const bool timing = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t start = timing ? scheduler_cycles() : 0;
    auto fail = [&](SchedulerErrorSite site) __aicore__ {
        scheduler_record_error(run, producer_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local, site);
        return false;
    };
    const auto *offsets = scheduler_state_at<uint64_t>(base, dependencies.fanout_offsets_offset);
    const uint64_t begin = offsets[producer_ref], end = offsets[producer_ref + 1];
    const auto *refs = scheduler_state_at<int32_t>(base, dependencies.fanout_refs_offset);
    auto *remaining = scheduler_state_at<uint64_t>(base, dependencies.remaining_offset);
    // TaskDone and successor counters occupy different atomic lines. A credit
    // may make the successor runnable on another core immediately.
    if (begin != end) scheduler_cache_barrier();
    SchedulerReadyPublication publication{};
    for (uint64_t edge = begin; edge < end; ++edge) {
        if (edge != begin && (edge - begin) % 32 == 0 && scheduler_read_worker_error(run, local->worker_id()) != 0)
            return false;
        const int64_t ref = refs[edge];
        const uint64_t previous = scheduler_gm_fetch_add(remaining[ref], UINT64_MAX);
        if (previous == 0 || previous > static_cast<uint64_t>(ref))
            return fail(SchedulerErrorSite::COMPLETION_INVALID_WAITER);
        if (previous != 1) continue;
        auto appended = scheduler_ready_publication_append(
            graph, base, local, ref, publication, direct_ready, direct_core_type, admission_owner
        );
        if (appended == SchedulerReadyAppendResult::FULL) {
            if (!scheduler_flush_ready_publication(graph, base, local, run, producer_ref, publication)) return false;
            appended = scheduler_ready_publication_append(
                graph, base, local, ref, publication, direct_ready, direct_core_type, admission_owner
            );
        }
        if (appended == SchedulerReadyAppendResult::DIRECT) continue;
        if (appended != SchedulerReadyAppendResult::ADDED)
            return fail(SchedulerErrorSite::COMPLETION_READY_APPEND_FAILED);
        if (publication.count == SCHEDULER_READY_PUBLICATION_CAPACITY &&
            !scheduler_flush_ready_publication(graph, base, local, run, producer_ref, publication))
            return false;
    }
    if (!scheduler_flush_ready_publication(graph, base, local, run, producer_ref, publication)) return false;
    if (timing) {
        auto *control = scheduler_task_control_at(base, local, producer_ref);
        // A synchronous CSR owner is the sole writer of this timing line.
        // Host consumes it after all cores have acknowledged retirement.
        control->completion_resolve_start_cycles = start;
        control->completion_resolve_end_cycles = scheduler_cycles();
        control->scheduler_worker_id = local->worker_id();
        control->completion_resolve_loop_iter = local->profiling->loop_iter;
        scheduler_publish_cache_line(&control->completion_resolve_start_cycles);
    }
    return true;
}

// The unique TaskDone owner has checked cancellation and retains this traversal
// on the current core. Dynamic heads/cursors still require atomic ownership and
// bounds checks. The immutable fanin view and detached next link survive a Ready
// prefix flush; neither requires another GM read before routing this waiter.
static __aicore__ __attribute__((noinline)) bool scheduler_resolve_wake_completion(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t producer_ref, uint64_t profiling_level, SchedulerReadyClaim *direct_ready, uint32_t direct_core_type,
    bool admission_owner
) {
    SchedulerScalarScope scalar_scope(local, SchedPhaseKind::ScalarResolve, producer_ref);
    const bool timing = scheduler_phase_timing_enabled(profiling_level);
    const uint64_t start = timing ? scheduler_cycles() : 0;
    auto *producer = scheduler_task_control_at(base, local, producer_ref);
    int64_t waiter_ref = scheduler_gm_exchange(producer->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
    if (waiter_ref < SCHEDULER_WAKE_LIST_OPEN) {
        scheduler_record_error(
            run, producer_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
            SchedulerErrorSite::COMPLETION_WAKE_ALREADY_CLOSED
        );
        return false;
    }
    uint32_t visited = 0;
    SchedulerReadyPublication publication{};
    for (; waiter_ref != SCHEDULER_WAKE_LIST_OPEN; ++visited) {
        if (visited != 0 && visited % 32 == 0 && scheduler_read_worker_error(run, local->worker_id()) != 0)
            return false;
        SchedulerFaninView fanin{};
        if (visited >= graph.task_count ||
            scheduler_node_fanin_view(graph, waiter_ref, &fanin) != SchedulerGraphResult::OK) {
            scheduler_record_error(
                run, producer_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                SchedulerErrorSite::COMPLETION_INVALID_WAITER
            );
            return false;
        }
        const uint64_t wake =
            scheduler_gm_fetch_add(scheduler_task_control_at(base, local, waiter_ref)->wake_cursor, UINT64_C(0));
        const uint32_t next_fanin = scheduler_wake_cursor_fanin(wake);
        const int64_t next_waiter = scheduler_wake_cursor_next(wake);
        if (next_waiter < SCHEDULER_WAKE_LIST_OPEN || next_waiter == waiter_ref || next_fanin == 0 ||
            next_fanin > fanin.count) {
            scheduler_record_error(
                run, producer_ref, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                SchedulerErrorSite::COMPLETION_INVALID_WAITER
            );
            return false;
        }
        if (next_fanin != fanin.count) {
            if (!scheduler_flush_ready_publication(graph, base, local, run, producer_ref, publication)) return false;
            const auto routed = scheduler_route_detached_waiter(graph, base, local, run, waiter_ref, fanin, next_fanin);
            if (routed == SchedulerRouteResult::ERROR) return false;
            if (routed == SchedulerRouteResult::WAITING) {
                waiter_ref = next_waiter;
                continue;
            }
        }
        auto appended = scheduler_ready_publication_append(
            graph, base, local, waiter_ref, publication, direct_ready, direct_core_type, admission_owner
        );
        if (appended == SchedulerReadyAppendResult::FULL) {
            if (!scheduler_flush_ready_publication(graph, base, local, run, producer_ref, publication)) return false;
            appended = scheduler_ready_publication_append(
                graph, base, local, waiter_ref, publication, direct_ready, direct_core_type, admission_owner
            );
        }
        if (appended != SchedulerReadyAppendResult::DIRECT && appended != SchedulerReadyAppendResult::ADDED)
            return scheduler_fail_ready_append(graph, base, local, run, producer_ref, waiter_ref);
        if (publication.count == SCHEDULER_READY_PUBLICATION_CAPACITY &&
            !scheduler_flush_ready_publication(graph, base, local, run, producer_ref, publication))
            return false;
        waiter_ref = next_waiter;
    }
    if (!scheduler_flush_ready_publication(graph, base, local, run, producer_ref, publication)) return false;
    if (timing) {
        producer->completion_resolve_start_cycles = start;
        producer->completion_resolve_end_cycles = scheduler_cycles();
        producer->scheduler_worker_id = local->worker_id();
        producer->completion_resolve_loop_iter = local->profiling->loop_iter;
        scheduler_publish_cache_line(&producer->completion_resolve_start_cycles);
    }
    return true;
}
