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

// AICore-resident dispatch library; scheduler_dispatch.cpp is the independent AICPU scheduler implementation.

#include "scheduler_completion.h"

struct SchedulerDeferredAivDispatch {
    SchedulerReadyClaim ready{};
    SchedulerFreeSlotClaim reserved_slot{};
};

struct SchedulerDeferredAivQueue {
    // Every entry owns one Scheduler slot reserved in local FILLING state, so
    // a peer miss can fall back to local execution without another capacity decision.
    SchedulerDeferredAivDispatch entries[SCHEDULER_PENDING_SLOT_COUNT]{};
    uint32_t count{0};
};

inline __aicore__ void scheduler_deferred_aiv_pop_front(SchedulerDeferredAivQueue *queue) {
    if (queue == nullptr || queue->count == 0) return;
    for (uint32_t index = 1; index < queue->count; ++index)
        queue->entries[index - 1] = queue->entries[index];
    --queue->count;
    queue->entries[queue->count] = {};
}

inline __aicore__ bool scheduler_normal_aiv_worker_precedes(
    uint32_t candidate_occupied_slots, bool candidate_is_scheduler, uint32_t selected_occupied_slots,
    bool selected_is_scheduler
) {
    if (candidate_is_scheduler != selected_is_scheduler) return !candidate_is_scheduler;
    return candidate_occupied_slots < selected_occupied_slots;
}

// The return value reports whether this pass made progress; failed independently reports an aborted pass.
inline __aicore__ bool scheduler_fill_cluster_normal_slots(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, uint64_t *ready_victim_cursors, SchedulerReadyStats *ready_stats,
    uint64_t profiling_level, uint64_t skip_slot_mask, SchedulerDeferredAivQueue *deferred_aiv, bool *failed,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region
) {
    if (failed != nullptr) *failed = false;
    if (scheduler == nullptr || !scheduler->is_scheduler()) return false;
    const uint32_t aic_core_type = static_cast<uint32_t>(CoreType::AIC);
    uint64_t state_probe_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    bool progress = false;

    // AIC has no peer lane in its Cluster, so preserve the existing slot order.
    if (scheduler_ready_directory_nonempty(
            scheduler_state_base, scheduler, scheduler->config.scheduler_count, aic_core_type
        )) {
        bool aic_ready_available = true;
        for (uint32_t cluster_lane = 0; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM && aic_ready_available;
             ++cluster_lane) {
            const uint64_t worker_id = scheduler->config.worker_ids[cluster_lane];
            if (worker_id >= scheduler->config.runtime_worker_count) continue;
            __gm__ SchedulerWorkerContext *target =
                scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
            if (target->active == 0 || target->core_type != static_cast<int32_t>(CoreType::AIC)) continue;
            for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
                if ((skip_slot_mask & (UINT64_C(1) << (cluster_lane * SCHEDULER_PENDING_SLOT_COUNT + pending_slot))) !=
                    0)
                    continue;
                SchedulerLocalSlotState *local_slot = &scheduler->slots[cluster_lane][pending_slot];
                if (local_slot->state != SchedulerDispatchSlotState::FREE) continue;
                SchedulerReadyClaim ready{};
                if (!scheduler_claim_ready_for_slot(
                        graph, scheduler_state_base, scheduler, run_control, scheduler->config.scheduler_count,
                        aic_core_type, &ready_victim_cursors[aic_core_type], ready_stats, &ready
                    )) {
                    if (failed != nullptr) *failed = true;
                    return progress;
                }
                if (ready.task_id < 0) {
                    aic_ready_available = false;
                    break;
                }
                ready.state_probe_start_cycles = state_probe_start_cycles;
                ready.state_probe_end_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
                SchedulerFreeSlotClaim claim{
                    worker_id,
                    pending_slot,
                    local_slot->generation,
                    cluster_lane,
                };
                local_slot->state = SchedulerDispatchSlotState::FILLING;
                if (!scheduler_fill_dispatch_slot(
                        graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level, ssbuf_region
                    )) {
                    if (failed != nullptr) *failed = true;
                    return progress;
                }
                progress = true;
                state_probe_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
            }
        }
    }
    // A Scheduler shares its AIV with Executor work. Exhaust the non-Scheduler
    // peer's free slots first, then claim more work only against reserved
    // Scheduler capacity. The caller decides the reserved work's owner after
    // the rest of this scheduling round completes.
    struct AivWorkerSlots {
        uint64_t worker_id{UINT64_MAX};
        uint32_t cluster_lane{UINT32_MAX};
        uint32_t free_mask{0};
        uint32_t occupied_slots{0};
        bool is_scheduler{false};
    };
    const uint32_t aiv_core_type = static_cast<uint32_t>(CoreType::AIV);
    state_probe_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
    if (scheduler_ready_directory_nonempty(
            scheduler_state_base, scheduler, scheduler->config.scheduler_count, aiv_core_type
        )) {
        AivWorkerSlots aiv_workers[PLATFORM_AIV_CORES_PER_BLOCKDIM]{};
        uint32_t aiv_worker_count = 0;
        for (uint32_t cluster_lane = 0; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM; ++cluster_lane) {
            const uint64_t worker_id = scheduler->config.worker_ids[cluster_lane];
            if (worker_id >= scheduler->config.runtime_worker_count) continue;
            __gm__ SchedulerWorkerContext *target =
                scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
            if (target->active == 0 || target->core_type != static_cast<int32_t>(CoreType::AIV)) continue;
            if (aiv_worker_count >= PLATFORM_AIV_CORES_PER_BLOCKDIM) {
                scheduler_record_error(
                    run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
                    SchedulerErrorSite::NORMAL_DISPATCH_INVALID_TOPOLOGY
                );
                if (failed != nullptr) *failed = true;
                return progress;
            }
            AivWorkerSlots &worker = aiv_workers[aiv_worker_count++];
            worker.worker_id = worker_id;
            worker.cluster_lane = cluster_lane;
            worker.is_scheduler = worker_id == scheduler->worker_id();
            for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
                if ((skip_slot_mask & (UINT64_C(1) << (cluster_lane * SCHEDULER_PENDING_SLOT_COUNT + pending_slot))) !=
                    0) {
                    ++worker.occupied_slots;
                    continue;
                }
                SchedulerLocalSlotState *local_slot = &scheduler->slots[cluster_lane][pending_slot];
                if (local_slot->state == SchedulerDispatchSlotState::FREE) worker.free_mask |= 1U << pending_slot;
                else ++worker.occupied_slots;
            }
        }
        for (uint32_t attempt = 0; attempt < aiv_worker_count * SCHEDULER_PENDING_SLOT_COUNT; ++attempt) {
            uint32_t selected = aiv_worker_count;
            for (uint32_t worker_index = 0; worker_index < aiv_worker_count; ++worker_index) {
                const AivWorkerSlots &candidate = aiv_workers[worker_index];
                if (candidate.free_mask == 0) continue;
                if (selected == aiv_worker_count ||
                    scheduler_normal_aiv_worker_precedes(
                        candidate.occupied_slots, candidate.is_scheduler, aiv_workers[selected].occupied_slots,
                        aiv_workers[selected].is_scheduler
                    ))
                    selected = worker_index;
            }
            if (selected == aiv_worker_count) break;
            AivWorkerSlots &worker = aiv_workers[selected];
            const uint32_t pending_slot = static_cast<uint32_t>(__builtin_ctz(worker.free_mask));
            worker.free_mask &= ~(1U << pending_slot);
            if (worker.is_scheduler && (deferred_aiv == nullptr || deferred_aiv->count >= SCHEDULER_PENDING_SLOT_COUNT))
                break;
            SchedulerLocalSlotState *local_slot = &scheduler->slots[worker.cluster_lane][pending_slot];
            SchedulerFreeSlotClaim claim{
                worker.worker_id,
                pending_slot,
                local_slot->generation,
                worker.cluster_lane,
            };
            local_slot->state = SchedulerDispatchSlotState::FILLING;
            SchedulerReadyClaim ready{};
            if (!scheduler_claim_ready_for_slot(
                    graph, scheduler_state_base, scheduler, run_control, scheduler->config.scheduler_count,
                    aiv_core_type, &ready_victim_cursors[aiv_core_type], ready_stats, &ready
                )) {
                local_slot->state = SchedulerDispatchSlotState::FREE;
                if (failed != nullptr) *failed = true;
                return progress;
            }
            if (ready.task_id < 0) {
                local_slot->state = SchedulerDispatchSlotState::FREE;
                break;
            }
            ready.state_probe_start_cycles = state_probe_start_cycles;
            if (worker.is_scheduler) {
                SchedulerDeferredAivDispatch &entry = deferred_aiv->entries[deferred_aiv->count++];
                entry = {ready, claim};
                ++worker.occupied_slots;
                progress = true;
                state_probe_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
                entry.ready.state_probe_end_cycles = state_probe_start_cycles;
                continue;
            }
            ready.state_probe_end_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
            if (!scheduler_fill_dispatch_slot(
                    graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level, ssbuf_region
                )) {
                if (failed != nullptr) *failed = true;
                return progress;
            }
            ++worker.occupied_slots;
            progress = true;
            state_probe_start_cycles = scheduler_phase_timing_enabled(profiling_level) ? scheduler_cycles() : 0;
        }
    }
    return progress;
}

inline __aicore__ bool scheduler_release_deferred_aiv_reservation(
    const SchedulerGraphView &graph, SchedulerLocalState *scheduler, __gm__ SchedulerRunControl *run_control,
    const SchedulerFreeSlotClaim &reservation
) {
    if (reservation.worker_id != scheduler->worker_id() || reservation.slot_index >= SCHEDULER_PENDING_SLOT_COUNT ||
        reservation.cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM) {
        scheduler_record_error(
            run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::DEFERRED_RESERVATION_INVALID_OWNER
        );
        return false;
    }
    SchedulerLocalSlotState *local_slot = &scheduler->slots[reservation.cluster_lane][reservation.slot_index];
    const int64_t task_id = local_slot->task_id;
    if (local_slot->state != SchedulerDispatchSlotState::FILLING || local_slot->generation != reservation.generation ||
        task_id != SCHEDULER_TASK_ID_INVALID) {
        scheduler_record_error(
            run_control, task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::DEFERRED_RESERVATION_INVALID_STATE
        );
        return false;
    }
    local_slot->state = SchedulerDispatchSlotState::FREE;
    return true;
}

inline __aicore__ int32_t
scheduler_deferred_aiv_peer_lane(__gm__ void *scheduler_state_base, SchedulerLocalState *scheduler) {
    for (uint32_t cluster_lane = 0; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM; ++cluster_lane) {
        const uint64_t worker_id = scheduler->config.worker_ids[cluster_lane];
        if (worker_id >= scheduler->config.runtime_worker_count) continue;
        if (worker_id == scheduler->worker_id()) continue;
        __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
        scheduler_observe_cache_line(target);
        if (target->active != 0 && target->core_type == static_cast<int32_t>(CoreType::AIV))
            return static_cast<int32_t>(cluster_lane);
    }
    return -1;
}

inline __aicore__ bool scheduler_drain_deferred_aiv_to_peer(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, SchedulerDeferredAivQueue *queue, SchedulerWakeStats *wake_stats,
    SchedulerReadyStats *ready_stats, SchedulerCompletionStats *completion_stats, uint64_t profiling_level,
    SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region
) {
    if (queue == nullptr || queue->count == 0) return true;
    if (ssbuf_region == nullptr || scheduler == nullptr) return false;
    const int32_t peer_lane = scheduler_deferred_aiv_peer_lane(scheduler_state_base, scheduler);
    // A Scheduler may be the only active AIV in its Cluster (for example, a
    // single-root AIV graph). There is then nothing to drain to; leave the
    // reservation queued so the caller can publish it on the Scheduler itself.
    if (peer_lane < 0) return true;
    const uint64_t peer_worker_id = scheduler->config.worker_ids[static_cast<uint32_t>(peer_lane)];
    for (uint32_t pass = 0; pass < 2 && queue->count != 0; ++pass) {
        const uint64_t publication =
            pass == 1 ? scheduler_ssbuf_load_relaxed(
                            &ssbuf_region->lanes[static_cast<uint32_t>(peer_lane)].completion.publication
                        ) :
                        0;
        bool acquired = false;
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT && queue->count != 0;
             ++pending_slot) {
            SchedulerLocalSlotState *local_slot = &scheduler->slots[static_cast<uint32_t>(peer_lane)][pending_slot];
            bool refilled = false;
            if (pass == 0) {
                if (local_slot->state != SchedulerDispatchSlotState::FREE) continue;
                local_slot->state = SchedulerDispatchSlotState::FILLING;
                if (!scheduler_fill_dispatch_slot(
                        graph, scheduler_state_base, scheduler, run_control,
                        SchedulerFreeSlotClaim{
                            peer_worker_id, pending_slot, local_slot->generation, static_cast<uint32_t>(peer_lane)
                        },
                        queue->entries[0].ready, profiling_level, ssbuf_region
                    ))
                    return false;
                refilled = true;
            } else {
                if (local_slot->state != SchedulerDispatchSlotState::READY) continue;
                const uint32_t completed_generation = static_cast<uint32_t>(publication >> (pending_slot * 32));
                if (completed_generation != local_slot->generation ||
                    !scheduler_completion_generation_is_new(
                        scheduler, static_cast<uint32_t>(peer_lane), pending_slot, completed_generation
                    ))
                    continue;
                if (!acquired) {
                    scheduler_cache_barrier();
                    acquired = true;
                }
                if (!scheduler_service_cluster_completion_slot(
                        graph, scheduler_state_base, scheduler, run_control, static_cast<uint32_t>(peer_lane),
                        pending_slot, completed_generation, wake_stats, ready_stats, completion_stats, nullptr,
                        profiling_level, &queue->entries[0].ready, &refilled, ssbuf_region
                    ) ||
                    !refilled)
                    return false;
            }
            if (!scheduler_release_deferred_aiv_reservation(
                    graph, scheduler, run_control, queue->entries[0].reserved_slot
                ))
                return false;
            scheduler_deferred_aiv_pop_front(queue);
        }
    }
    return true;
}

inline __aicore__ bool scheduler_publish_deferred_aiv_local(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, SchedulerLocalState *scheduler,
    __gm__ SchedulerRunControl *run_control, SchedulerDeferredAivQueue *queue, uint64_t profiling_level,
    uint32_t *published_slot, SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf_region
) {
    if (published_slot != nullptr) *published_slot = UINT32_MAX;
    if (queue == nullptr || queue->count == 0) return true;
    if (ssbuf_region == nullptr || scheduler == nullptr) return false;
    const SchedulerDeferredAivDispatch &entry = queue->entries[0];
    if (entry.reserved_slot.worker_id != scheduler->worker_id() ||
        entry.reserved_slot.slot_index >= SCHEDULER_PENDING_SLOT_COUNT ||
        entry.reserved_slot.cluster_lane >= PLATFORM_CORES_PER_BLOCKDIM) {
        scheduler_record_error(
            run_control, entry.ready.task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::DEFERRED_PUBLISH_INVALID_RESERVATION
        );
        return false;
    }
    SchedulerLocalSlotState *local_slot =
        &scheduler->slots[entry.reserved_slot.cluster_lane][entry.reserved_slot.slot_index];
    const int64_t task_id = local_slot->task_id;
    if (local_slot->state != SchedulerDispatchSlotState::FILLING ||
        local_slot->generation != entry.reserved_slot.generation || task_id != SCHEDULER_TASK_ID_INVALID) {
        scheduler_record_error(
            run_control, entry.ready.task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::DEFERRED_PUBLISH_INVALID_RESERVATION
        );
        return false;
    }
    if (!scheduler_fill_dispatch_slot(
            graph, scheduler_state_base, scheduler, run_control, entry.reserved_slot, entry.ready, profiling_level,
            ssbuf_region
        ))
        return false;
    if (published_slot != nullptr) *published_slot = entry.reserved_slot.slot_index;
    scheduler_deferred_aiv_pop_front(queue);
    return true;
}
