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
    // Every entry owns one Scheduler slot held in FILLING, so a peer miss can
    // always fall back to local execution without another capacity decision.
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

struct SchedulerNormalDispatchTiming {
    uint64_t probe_cycles[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t claim_cycles[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t prepare_cycles[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t materialize_cycles[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t publish_cycles[SCHEDULER_CORE_TYPE_COUNT]{};
};

inline __aicore__ uint64_t
scheduler_normal_dispatch_detail_cycles(const SchedulerNormalDispatchTiming &timing, uint32_t core_type) {
    return timing.claim_cycles[core_type] + timing.prepare_cycles[core_type] + timing.materialize_cycles[core_type] +
           timing.publish_cycles[core_type];
}

inline __aicore__ void scheduler_finish_normal_dispatch_stage(
    SchedulerNormalDispatchTiming *timing, uint32_t core_type, uint64_t stage_start, uint64_t detail_start
) {
    if (timing == nullptr) return;
    uint64_t stage_cycles = scheduler_cycles() - stage_start;
    uint64_t detail_cycles = scheduler_normal_dispatch_detail_cycles(*timing, core_type) - detail_start;
    timing->probe_cycles[core_type] += stage_cycles > detail_cycles ? stage_cycles - detail_cycles : 0;
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
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, uint64_t *ready_victim_cursors, SchedulerReadyStats *ready_stats,
    uint64_t profiling_level, uint64_t skip_slot_mask = 0, SchedulerNormalDispatchTiming *timing = nullptr,
    SchedulerDeferredAivQueue *deferred_aiv = nullptr, __gm__ SchedulerReadyOwnerState *owner_state = nullptr,
    bool *failed = nullptr
) {
    if (failed != nullptr) *failed = false;
    if (scheduler->is_scheduler == 0) return false;
    const uint32_t aic_core_type = static_cast<uint32_t>(CoreType::AIC);
    uint64_t stage_start = timing == nullptr ? 0 : scheduler_cycles();
    uint64_t detail_start = timing == nullptr ? 0 : scheduler_normal_dispatch_detail_cycles(*timing, aic_core_type);
    bool progress = false;

    // AIC has no peer lane in its Cluster, so preserve the existing slot order.
    if (scheduler_ready_directory_nonempty(
            scheduler_state_base, scheduler, scheduler->scheduler_count, aic_core_type
        )) {
        bool aic_ready_available = true;
        for (uint32_t cluster_lane = 0; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM && aic_ready_available;
             ++cluster_lane) {
            const uint64_t worker_id = scheduler->cluster_worker_ids[cluster_lane];
            if (worker_id >= scheduler->runtime_worker_count) continue;
            __gm__ SchedulerWorkerContext *target =
                scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
            if (target->active == 0 || target->core_type != static_cast<int32_t>(CoreType::AIC)) continue;
            for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
                if ((skip_slot_mask & (UINT64_C(1) << (cluster_lane * SCHEDULER_PENDING_SLOT_COUNT + pending_slot))) !=
                    0)
                    continue;
                __gm__ SchedulerDispatchSlot *slot =
                    scheduler_dispatch_slot_at(scheduler_state_base, scheduler, worker_id, pending_slot);
                const uint64_t publication = scheduler_gm_query(slot->publication);
                if (scheduler_dispatch_state(publication) != SchedulerDispatchSlotState::FREE) continue;
                SchedulerReadyClaim ready{};
                if (!scheduler_claim_ready_for_slot(
                        graph, scheduler_state_base, scheduler, run_control, scheduler->scheduler_count, aic_core_type,
                        &ready_victim_cursors[aic_core_type], ready_stats, &ready, owner_state, profiling_level
                    )) {
                    if (failed != nullptr) *failed = true;
                    return progress;
                }
                if (timing != nullptr && ready.claim_end_cycles >= ready.claim_start_cycles)
                    timing->claim_cycles[aic_core_type] += ready.claim_end_cycles - ready.claim_start_cycles;
                if (ready.task_id < 0) {
                    aic_ready_available = false;
                    break;
                }
                SchedulerFreeSlotClaim claim{
                    worker_id,
                    pending_slot,
                    scheduler_dispatch_generation(publication),
                };
                scheduler_gm_store(
                    slot->publication,
                    scheduler_dispatch_publication(claim.generation, SchedulerDispatchSlotState::FILLING)
                );
                SchedulerDispatchFillTiming fill_timing{};
                if (!scheduler_fill_dispatch_slot(
                        graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level,
                        timing == nullptr ? nullptr : &fill_timing
                    )) {
                    if (failed != nullptr) *failed = true;
                    return progress;
                }
                if (timing != nullptr) {
                    timing->prepare_cycles[aic_core_type] += fill_timing.prepare_cycles;
                    timing->materialize_cycles[aic_core_type] += fill_timing.materialize_cycles;
                    timing->publish_cycles[aic_core_type] += fill_timing.publish_cycles;
                }
                progress = true;
            }
        }
    }
    scheduler_finish_normal_dispatch_stage(timing, aic_core_type, stage_start, detail_start);

    // A Scheduler shares its AIV with Executor work. Exhaust the non-Scheduler
    // peer's free slots first, then claim more work only against reserved
    // Scheduler capacity. The caller decides the reserved work's owner after
    // the rest of this scheduling round completes.
    struct AivWorkerSlots {
        uint64_t worker_id{UINT64_MAX};
        uint64_t publications[SCHEDULER_PENDING_SLOT_COUNT]{};
        uint32_t free_mask{0};
        uint32_t occupied_slots{0};
        bool is_scheduler{false};
    };
    const uint32_t aiv_core_type = static_cast<uint32_t>(CoreType::AIV);
    stage_start = timing == nullptr ? 0 : scheduler_cycles();
    detail_start = timing == nullptr ? 0 : scheduler_normal_dispatch_detail_cycles(*timing, aiv_core_type);
    if (scheduler_ready_directory_nonempty(
            scheduler_state_base, scheduler, scheduler->scheduler_count, aiv_core_type
        )) {
        AivWorkerSlots aiv_workers[PLATFORM_AIV_CORES_PER_BLOCKDIM]{};
        uint32_t aiv_worker_count = 0;
        for (uint32_t cluster_lane = 0; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM; ++cluster_lane) {
            const uint64_t worker_id = scheduler->cluster_worker_ids[cluster_lane];
            if (worker_id >= scheduler->runtime_worker_count) continue;
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
            worker.is_scheduler = worker_id == scheduler->worker_index;
            for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
                if ((skip_slot_mask & (UINT64_C(1) << (cluster_lane * SCHEDULER_PENDING_SLOT_COUNT + pending_slot))) !=
                    0) {
                    ++worker.occupied_slots;
                    continue;
                }
                __gm__ SchedulerDispatchSlot *slot =
                    scheduler_dispatch_slot_at(scheduler_state_base, scheduler, worker_id, pending_slot);
                const uint64_t publication = scheduler_gm_query(slot->publication);
                worker.publications[pending_slot] = publication;
                if (scheduler_dispatch_state(publication) == SchedulerDispatchSlotState::FREE)
                    worker.free_mask |= 1U << pending_slot;
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
            const uint64_t publication = worker.publications[pending_slot];
            SchedulerFreeSlotClaim claim{
                worker.worker_id,
                pending_slot,
                scheduler_dispatch_generation(publication),
            };
            __gm__ SchedulerDispatchSlot *slot =
                scheduler_dispatch_slot_at(scheduler_state_base, scheduler, worker.worker_id, pending_slot);
            scheduler_gm_store(
                slot->publication, scheduler_dispatch_publication(claim.generation, SchedulerDispatchSlotState::FILLING)
            );
            SchedulerReadyClaim ready{};
            if (!scheduler_claim_ready_for_slot(
                    graph, scheduler_state_base, scheduler, run_control, scheduler->scheduler_count, aiv_core_type,
                    &ready_victim_cursors[aiv_core_type], ready_stats, &ready, owner_state, profiling_level
                )) {
                scheduler_gm_store(
                    slot->publication,
                    scheduler_dispatch_publication(claim.generation, SchedulerDispatchSlotState::FREE)
                );
                if (failed != nullptr) *failed = true;
                return progress;
            }
            if (timing != nullptr && ready.claim_end_cycles >= ready.claim_start_cycles)
                timing->claim_cycles[aiv_core_type] += ready.claim_end_cycles - ready.claim_start_cycles;
            if (ready.task_id < 0) {
                scheduler_gm_store(
                    slot->publication,
                    scheduler_dispatch_publication(claim.generation, SchedulerDispatchSlotState::FREE)
                );
                break;
            }
            if (worker.is_scheduler) {
                deferred_aiv->entries[deferred_aiv->count++] = {ready, claim};
                ++worker.occupied_slots;
                progress = true;
                continue;
            }
            SchedulerDispatchFillTiming fill_timing{};
            if (!scheduler_fill_dispatch_slot(
                    graph, scheduler_state_base, scheduler, run_control, claim, ready, profiling_level,
                    timing == nullptr ? nullptr : &fill_timing
                )) {
                if (failed != nullptr) *failed = true;
                return progress;
            }
            if (timing != nullptr) {
                timing->prepare_cycles[aiv_core_type] += fill_timing.prepare_cycles;
                timing->materialize_cycles[aiv_core_type] += fill_timing.materialize_cycles;
                timing->publish_cycles[aiv_core_type] += fill_timing.publish_cycles;
            }
            ++worker.occupied_slots;
            progress = true;
        }
    }
    scheduler_finish_normal_dispatch_stage(timing, aiv_core_type, stage_start, detail_start);
    return progress;
}

inline __aicore__ bool scheduler_release_deferred_aiv_reservation(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, const SchedulerFreeSlotClaim &reservation
) {
    if (reservation.worker_id != scheduler->worker_index || reservation.slot_index >= SCHEDULER_PENDING_SLOT_COUNT) {
        scheduler_record_error(
            run_control, SCHEDULER_TASK_ID_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::DEFERRED_RESERVATION_INVALID_OWNER
        );
        return false;
    }
    __gm__ SchedulerDispatchSlot *slot =
        scheduler_dispatch_slot_at(scheduler_state_base, scheduler, reservation.worker_id, reservation.slot_index);
    const uint64_t publication = scheduler_gm_query(slot->publication);
    if (scheduler_dispatch_state(publication) != SchedulerDispatchSlotState::FILLING ||
        scheduler_dispatch_generation(publication) != reservation.generation ||
        slot->task_id != SCHEDULER_TASK_ID_INVALID) {
        scheduler_record_error(
            run_control, slot->task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::DEFERRED_RESERVATION_INVALID_STATE
        );
        return false;
    }
    scheduler_gm_publish(
        slot->publication, scheduler_dispatch_publication(reservation.generation, SchedulerDispatchSlotState::FREE)
    );
    return true;
}

inline __aicore__ int32_t
scheduler_deferred_aiv_peer_lane(__gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler) {
    for (uint32_t cluster_lane = 0; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM; ++cluster_lane) {
        const uint64_t worker_id = scheduler->cluster_worker_ids[cluster_lane];
        if (worker_id >= scheduler->runtime_worker_count) continue;
        if (worker_id == scheduler->worker_index) continue;
        __gm__ SchedulerWorkerContext *target = scheduler_worker_context_at(scheduler_state_base, scheduler, worker_id);
        scheduler_observe_cache_line(target);
        if (target->active != 0 && target->core_type == static_cast<int32_t>(CoreType::AIV))
            return static_cast<int32_t>(cluster_lane);
    }
    return -1;
}

inline __aicore__ bool scheduler_drain_deferred_aiv_to_peer(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, SchedulerDeferredAivQueue *queue, SchedulerWakeStats *wake_stats,
    SchedulerReadyStats *ready_stats, SchedulerCompletionStats *completion_stats, uint64_t profiling_level,
    SchedulerCompletionServiceTiming *completion_timing = nullptr,
    SchedulerNormalDispatchTiming *dispatch_timing = nullptr, __gm__ SchedulerReadyOwnerState *owner_state = nullptr
) {
    if (queue == nullptr || queue->count == 0) return true;
    const int32_t peer_lane = scheduler_deferred_aiv_peer_lane(scheduler_state_base, scheduler);
    // A Scheduler may be the only active AIV in its Cluster (for example, a
    // single-root AIV graph). There is then nothing to drain to; leave the
    // reservation queued so the caller can publish it on the Scheduler itself.
    if (peer_lane < 0) return true;
    const uint64_t peer_worker_id = scheduler->cluster_worker_ids[static_cast<uint32_t>(peer_lane)];
    __gm__ SchedulerCompletionInbox *completion_line =
        scheduler_completion_inbox_at(scheduler_state_base, scheduler, peer_worker_id);
    const uint32_t aiv_core_type = static_cast<uint32_t>(CoreType::AIV);

    for (uint32_t pass = 0; pass < 2 && queue->count != 0; ++pass) {
        const uint64_t completed_generations =
            pass == 1 ? scheduler_gm_query_u32_pair(completion_line->completed_generations) : 0;
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT && queue->count != 0;
             ++pending_slot) {
            __gm__ SchedulerDispatchSlot *peer_slot =
                scheduler_dispatch_slot_at(scheduler_state_base, scheduler, peer_worker_id, pending_slot);
            const uint64_t publication = scheduler_gm_query(peer_slot->publication);
            const SchedulerDispatchSlotState state = scheduler_dispatch_state(publication);
            const uint32_t generation = scheduler_dispatch_generation(publication);
            bool refilled = false;
            if (pass == 0) {
                if (state != SchedulerDispatchSlotState::FREE) continue;
                scheduler_gm_store(
                    peer_slot->publication,
                    scheduler_dispatch_publication(generation, SchedulerDispatchSlotState::FILLING)
                );
                SchedulerDispatchFillTiming fill_timing{};
                if (!scheduler_fill_dispatch_slot(
                        graph, scheduler_state_base, scheduler, run_control,
                        SchedulerFreeSlotClaim{peer_worker_id, pending_slot, generation}, queue->entries[0].ready,
                        profiling_level, dispatch_timing == nullptr ? nullptr : &fill_timing
                    ))
                    return false;
                if (dispatch_timing != nullptr) {
                    dispatch_timing->prepare_cycles[aiv_core_type] += fill_timing.prepare_cycles;
                    dispatch_timing->materialize_cycles[aiv_core_type] += fill_timing.materialize_cycles;
                    dispatch_timing->publish_cycles[aiv_core_type] += fill_timing.publish_cycles;
                }
                refilled = true;
            } else {
                if (state != SchedulerDispatchSlotState::READY) continue;
                const uint32_t completed_generation =
                    static_cast<uint32_t>(completed_generations >> (pending_slot * 32));
                if (completed_generation != generation) continue;
                scheduler_observe_cache_line(peer_slot);
                if (peer_slot->gang != 0) continue;
                if (!scheduler_service_cluster_completion_slot(
                        graph, scheduler_state_base, scheduler, run_control, static_cast<uint32_t>(peer_lane),
                        pending_slot, completed_generation, wake_stats, ready_stats, completion_stats, nullptr,
                        profiling_level, &queue->entries[0].ready, &refilled, completion_timing, owner_state
                    ) ||
                    !refilled)
                    return false;
            }
            if (!scheduler_release_deferred_aiv_reservation(
                    graph, scheduler_state_base, scheduler, run_control, queue->entries[0].reserved_slot
                ))
                return false;
            scheduler_deferred_aiv_pop_front(queue);
        }
    }
    return true;
}

inline __aicore__ bool scheduler_publish_deferred_aiv_local(
    const SchedulerGraphView &graph, __gm__ void *scheduler_state_base, __gm__ SchedulerWorkerContext *scheduler,
    __gm__ SchedulerRunControl *run_control, SchedulerDeferredAivQueue *queue, uint64_t profiling_level,
    uint32_t *published_slot, SchedulerNormalDispatchTiming *timing = nullptr
) {
    if (published_slot != nullptr) *published_slot = UINT32_MAX;
    if (queue == nullptr || queue->count == 0) return true;
    const SchedulerDeferredAivDispatch &entry = queue->entries[0];
    if (entry.reserved_slot.worker_id != scheduler->worker_index ||
        entry.reserved_slot.slot_index >= SCHEDULER_PENDING_SLOT_COUNT) {
        scheduler_record_error(
            run_control, entry.ready.task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::DEFERRED_PUBLISH_INVALID_RESERVATION
        );
        return false;
    }
    __gm__ SchedulerDispatchSlot *slot = scheduler_dispatch_slot_at(
        scheduler_state_base, scheduler, entry.reserved_slot.worker_id, entry.reserved_slot.slot_index
    );
    const uint64_t publication = scheduler_gm_query(slot->publication);
    scheduler_observe_cache_line(slot);
    if (scheduler_dispatch_state(publication) != SchedulerDispatchSlotState::FILLING ||
        scheduler_dispatch_generation(publication) != entry.reserved_slot.generation ||
        slot->task_id != SCHEDULER_TASK_ID_INVALID) {
        scheduler_record_error(
            run_control, entry.ready.task_id, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, scheduler,
            SchedulerErrorSite::DEFERRED_PUBLISH_INVALID_RESERVATION
        );
        return false;
    }
    const uint32_t aiv_core_type = static_cast<uint32_t>(CoreType::AIV);
    SchedulerDispatchFillTiming fill_timing{};
    if (!scheduler_fill_dispatch_slot(
            graph, scheduler_state_base, scheduler, run_control, entry.reserved_slot, entry.ready, profiling_level,
            timing == nullptr ? nullptr : &fill_timing
        ))
        return false;
    if (timing != nullptr) {
        timing->prepare_cycles[aiv_core_type] += fill_timing.prepare_cycles;
        timing->materialize_cycles[aiv_core_type] += fill_timing.materialize_cycles;
        timing->publish_cycles[aiv_core_type] += fill_timing.publish_cycles;
    }
    if (published_slot != nullptr) *published_slot = entry.reserved_slot.slot_index;
    scheduler_deferred_aiv_pop_front(queue);
    return true;
}
