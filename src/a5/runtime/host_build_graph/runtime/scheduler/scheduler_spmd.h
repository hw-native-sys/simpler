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

// Claiming blocks of an SPMD task. A block is taken from the head of the one
// ready queue the whole machine can see; a task of many blocks is therefore
// spread over the machine rather than confined to the seven cores that share
// its publisher's directory shard.
//
// Only one task is ever being claimed: the head itself, which does not move
// until its last block is taken. That is what makes "the claim did not fit"
// have exactly one answer -- give up this round -- rather than needing a rule
// for which other task to try, and it is also what keeps two sync-start cohorts
// from assembling at once without any token to arbitrate them.

#include <cstdint>

#include "common/platform_config.h"
#include "scheduler_graph.h"
#include "scheduler_memory.h"
#include "scheduler_mix.h"
#include "scheduler_ready.h"
#include "scheduler_ssbuf.h"
#include "scheduler_types.h"

// What this core may do with the SPMD head it is looking at.
enum class SchedulerSpmdHeadVerdict : uint8_t {
    // Nothing queued, so nothing to look at.
    EMPTY = 0,
    // Queued, but this core cannot run this task's shape.
    NOT_MINE = 1,
    // This core may take a block of it.
    CLAIMABLE = 2,
};

// Read the SPMD queue head and the shape facts a claim needs, reusing this
// core's memo when the head has not moved.
//
// The memo is what keeps the steady-state poll at one ld_dev. Without it every
// core pays a metadata cache-line observe on every pass for as long as one task
// holds the head, and the cores whose type does not match the head pay it while
// getting nothing -- for an AIC-only task of many blocks, that is 72 AIV cores
// for the duration.
inline __aicore__ bool scheduler_observe_spmd_head(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerSpmdHeadVerdict *verdict
) {
    *verdict = SchedulerSpmdHeadVerdict::EMPTY;
    __gm__ SchedulerReadyInbox *inbox = scheduler_ready_inbox_at(base, local, TaskType::SPMD, 0);
    const uint64_t packed = scheduler_gm_query(inbox->endpoints);
    if (!scheduler_ready_endpoints_consistent(packed)) {
        scheduler_record_error(
            run, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
            SchedulerErrorSite::READY_POP_INVALID_HEAD
        );
        return false;
    }
    if (scheduler_ready_endpoints_empty(packed)) {
        local->spmd_head_ref = SCHEDULER_TASK_REF_INVALID;
        return true;
    }
    const int64_t head = scheduler_ready_endpoints_head(packed);
    if (!scheduler_task_ref_valid(graph, head)) {
        scheduler_record_error(
            run, head, SchedulerGraphResult::INVALID_TASK_ID, &graph, local, SchedulerErrorSite::READY_POP_INVALID_HEAD
        );
        return false;
    }
    if (head != local->spmd_head_ref) {
        __gm__ SchedulerTaskMetadata *metadata = scheduler_task_metadata_at(base, local, head);
        scheduler_observe_cache_line(metadata);
        __gm__ SchedulerTaskControl *control = scheduler_task_control_at(base, local, head);
        // The host writes this once per run from the graph and never revisits
        // it, and it rejects a count outside [1, UINT16_MAX] before the graph
        // reaches the resident scheduler at all -- so memoizing it alongside
        // the shape costs nothing in freshness and needs no range check here.
        local->spmd_head_ref = head;
        local->spmd_head_active_mask = metadata->active_mask;
        local->spmd_head_flags = metadata->flags;
        local->spmd_head_block_num = static_cast<uint16_t>(scheduler_gm_query(control->block_num));
    }
    const bool mine = scheduler_task_is_mix(local->spmd_head_flags) ?
                          local->is_mix_coordinator() :
                          scheduler_shape_runs_on_core_type(local->spmd_head_active_mask, local->config.self_core_type);
    *verdict = mine ? SchedulerSpmdHeadVerdict::CLAIMABLE : SchedulerSpmdHeadVerdict::NOT_MINE;
    return true;
}

// Let the next SPMD task become the head, but only while the head is still the
// task the caller found exhausted.
//
// The identity check is the whole of it. An ordinary pop retries by re-reading
// the endpoints, so two cores that both decide the head owes nothing take two
// tasks: the first pops the exhausted one, the second re-reads, finds its
// successor, and pops that -- with every one of its blocks still unclaimed and
// no queue left to find it in. The CAS alone does not prevent this; it makes
// the update atomic, not the decision. Both cores reaching here at once is
// ordinary, because the claim loop ends with `current >= block_num` on every
// core that was racing for the last block.
//
// Not the shared pop for a second reason: that one takes a placement and
// leaves a task queued when it cannot be placed. A Mix head whose last block
// was just taken has its own blocks in the cluster's Mix slots, so placement
// must fail -- and the head would never move.
inline __aicore__ bool scheduler_advance_spmd_head(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    int64_t exhausted, SchedulerReadyStats *stats
) {
    __gm__ SchedulerReadyInbox *inbox = scheduler_ready_inbox_at(base, local, TaskType::SPMD, 0);
    for (uint32_t attempt = 0; attempt < 64; ++attempt) {
        const uint64_t packed = scheduler_gm_query(inbox->endpoints);
        if (!scheduler_ready_endpoints_consistent(packed)) {
            scheduler_record_error(
                run, exhausted, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                SchedulerErrorSite::READY_POP_INVALID_HEAD
            );
            return false;
        }
        if (scheduler_ready_endpoints_empty(packed)) break;
        const int64_t head = scheduler_ready_endpoints_head(packed);
        // Already advanced, by whichever core got here first.
        if (head != exhausted) break;
        const int64_t tail = scheduler_ready_endpoints_tail(packed);
        if (!scheduler_task_ref_valid(graph, tail)) {
            scheduler_record_error(
                run, tail, SchedulerGraphResult::INVALID_TASK_ID, &graph, local,
                SchedulerErrorSite::READY_POP_INVALID_HEAD
            );
            return false;
        }
        int64_t next = SCHEDULER_INBOX_EMPTY;
        if (head != tail) {
            __gm__ SchedulerTaskControl *control = scheduler_task_control_at(base, local, head);
            for (uint32_t spin = 0; spin < SCHEDULER_READY_LINK_SPIN_LIMIT; ++spin) {
                next = scheduler_observe_next_waiter(control);
                if (next != SCHEDULER_INBOX_EMPTY) break;
            }
            if (next < SCHEDULER_INBOX_EMPTY) {
                scheduler_record_error(
                    run, head, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, local,
                    SchedulerErrorSite::READY_POP_INVALID_LINK
                );
                return false;
            }
            // A push stores the link a few instructions after its CAS, so an
            // unpublished one means re-read rather than end-of-queue.
            if (next == SCHEDULER_INBOX_EMPTY) continue;
            if (!scheduler_task_ref_valid(graph, next)) {
                scheduler_record_error(
                    run, next, SchedulerGraphResult::INVALID_TASK_ID, &graph, local,
                    SchedulerErrorSite::READY_POP_INVALID_HEAD
                );
                return false;
            }
        }
        const uint64_t next_packed = next == SCHEDULER_INBOX_EMPTY ? SCHEDULER_READY_ENDPOINTS_EMPTY :
                                                                     scheduler_ready_endpoints_pack(next, tail);
        if (scheduler_gm_compare_exchange(inbox->endpoints, packed, next_packed) == packed) {
            if (stats != nullptr) ++stats->pop_count;
            break;
        }
        if (stats != nullptr) ++stats->cas_retry_count;
    }
    // Whoever advanced it, this core's memo now describes a task that is no
    // longer the head.
    local->spmd_head_ref = SCHEDULER_TASK_REF_INVALID;
    return true;
}

// Take one block of the head for this core, if the head is one this core can
// run and there is somewhere to put the block.
//
// One block per round, deliberately. Taking several would cut CAS traffic but
// let whichever core frees up first carry off the work, leaving the cores that
// free up later with nothing: contention is tolerable, imbalance is not.
inline __aicore__ bool scheduler_claim_spmd_block(
    const SchedulerGraphView &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerReadyStats *stats, uint64_t profiling_level, SCHEDULER_SSBUF SchedulerSsbufRegion *ssbuf, bool *progress
) {
    if (!local->has_spmd) return true;
    SchedulerSpmdHeadVerdict verdict = SchedulerSpmdHeadVerdict::EMPTY;
    if (!scheduler_observe_spmd_head(graph, base, local, run, &verdict)) return false;
    if (verdict != SchedulerSpmdHeadVerdict::CLAIMABLE) return true;

    const int64_t task_ref = local->spmd_head_ref;
    const uint16_t block_num = local->spmd_head_block_num;
    const bool is_mix = scheduler_task_is_mix(local->spmd_head_flags);

    // A cohort's blocks wait on each other inside the kernel, so two of them on
    // one core self-lock: the second sits in a dispatch slot the first cannot
    // release, because the first is spinning in a barrier and its core's whole
    // scheduling loop is stopped with it. Scanning for the task is an exact
    // test that needs no shared state and expires on its own, since a block
    // holds its slot until it completes -- and the question is purely local,
    // so no peer has to be told the answer.
    //
    // The Mix slots are scanned only because a Mix task need not have an AIC
    // subtask: the flags say MIX whenever more than one subtask is live, which
    // includes two AIVs and no AIC. A Mix block with an AIC subtask always puts
    // it in the coordinator's own slots, so the first loop already finds it;
    // an AIV-only one lands entirely on Mix slots and would otherwise be missed.
    // Only the coordinator can run either test for a Mix cohort anyway: an AIV
    // executing a Mix slot reads its task from SSBUF and holds no slot state
    // for it, so it cannot see what it is already running.
    if (scheduler_task_requires_sync_start(local->spmd_head_flags)) {
        for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot)
            if (local->own_slots()[slot].task_ref == task_ref) return true;
        if (is_mix) {
            // Lane 0 is the coordinator itself and holds no Mix slot, so the
            // live ones are the two AIV lanes' two slots each.
            for (uint32_t lane = 1; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
                for (uint32_t slot = 0; slot < SCHEDULER_MIX_SLOT_COUNT; ++slot)
                    if (local->mix_slots[lane][slot].task_ref == task_ref) return true;
        }
    }

    // Probe placement before taking a block, never after. The cursor cannot be
    // put back, so a block claimed with nowhere to run is lost for the whole
    // run and the task's completed count never reaches its total. A probe
    // changes no shared state, so failing it costs nothing to undo -- and with
    // one task at the head there is no second task to try instead.
    SchedulerMixPlacement placement{};
    uint32_t own_slot = UINT32_MAX;
    if (is_mix) {
        if (!scheduler_plan_mix_placement(base, local, task_ref, &placement)) return true;
    } else {
        for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot) {
            if (local->own_slots()[slot].state != SchedulerDispatchSlotState::FREE) continue;
            own_slot = slot;
            break;
        }
        if (own_slot == UINT32_MAX) return true;
    }

    __gm__ SchedulerTaskControl *control = scheduler_task_control_at(base, local, task_ref);
    // This first read is the only one that is not an atomic, and a device load
    // does not observe what an atomic wrote until that write has drained, so it
    // may answer with a value the cursor has already passed. Two things make
    // that harmless, and the second is what the loop below rests on:
    //
    //   - too small, and the CAS fails and hands back the current value;
    //   - never too large, because the cursor only ever moves forward. If it
    //     were possible to read a value above the true one, this core could
    //     skip the loop and go on to yield a head that still had blocks to
    //     give -- the same loss the identity check in the advance guards
    //     against, reached from the other side.
    uint64_t current = scheduler_gm_query(control->next_block);
    uint64_t claimed = UINT64_MAX;
    while (current < block_num) {
        const uint64_t observed = scheduler_gm_compare_exchange(control->next_block, current, current + 1);
        if (observed == current) {
            claimed = current;
            break;
        }
        if (stats != nullptr) ++stats->cas_retry_count;
        current = observed;
    }
    if (claimed == UINT64_MAX) {
        // Claimed clean by other cores while this one was probing. The head
        // owes nothing further, so let the next SPMD task have it.
        return scheduler_advance_spmd_head(graph, base, local, run, task_ref, stats);
    }

    SchedulerReadyClaim ready{};
    ready.task_ref = task_ref;
    ready.inbox_index = 0;
    ready.source = SchedulerReadySource::SPMD_CLAIM;
    const uint16_t block_idx = static_cast<uint16_t>(claimed);
    bool dispatched = false;
    if (is_mix) {
        // The three subtasks are parts of one block, so they share its index.
        for (uint8_t i = 0; i < placement.count; ++i) {
            placement.slots[i].block_idx = block_idx;
            placement.slots[i].block_num = block_num;
        }
        dispatched = scheduler_dispatch_mix_ready(graph, base, local, run, ready, placement, profiling_level, ssbuf);
    } else {
        SchedulerLocalSlotState *slot_state = &local->own_slots()[own_slot];
        SchedulerFreeSlotClaim claim{local->worker_id(), own_slot, slot_state->generation, local->config.self_lane};
        claim.block_idx = block_idx;
        claim.block_num = block_num;
        slot_state->state = SchedulerDispatchSlotState::FILLING;
        dispatched = scheduler_fill_dispatch_slot(graph, base, local, run, claim, ready, profiling_level, ssbuf);
    }
    if (!dispatched) return false;
    if (progress != nullptr) *progress = true;
    if (stats != nullptr) ++stats->pop_count;

    if (claimed + 1 == block_num) return scheduler_advance_spmd_head(graph, base, local, run, task_ref, stats);
    return true;
}
