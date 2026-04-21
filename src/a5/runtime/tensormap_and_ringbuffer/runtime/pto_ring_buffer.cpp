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
/**
 * PTO Runtime2 - Ring Buffer Implementation
 *
 * Implements DepListPool ring buffer for zero-overhead dependency management.
 * TaskAllocator methods are defined inline in pto_ring_buffer.h.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_ring_buffer.h"
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>  // for exit()
#include "common/unified_log.h"
#include "pto_scheduler.h"

// =============================================================================
// Fanin Spill Pool Implementation
// =============================================================================
void PTO2FaninPool::reclaim(PTO2SharedMemoryHandle &sm_handle, uint8_t ring_id, int32_t sm_last_task_alive) {
    if (sm_last_task_alive <= reclaim_task_cursor) return;

    int32_t scan_end = sm_last_task_alive;
    for (int32_t task_id = reclaim_task_cursor; task_id < scan_end; ++task_id) {
        PTO2TaskSlotState &slot_state = sm_handle.get_slot_state_by_task_id(ring_id, task_id);
        PTO2TaskPayload *payload = slot_state.payload;
        if (payload == nullptr || payload->fanin_spill_pool != this) {
            continue;
        }

        int32_t inline_count = std::min(payload->fanin_actual_count, PTO2_FANIN_INLINE_CAP);
        int32_t spill_edge_count = payload->fanin_actual_count - inline_count;
        if (spill_edge_count > 0) {
            advance_tail(payload->fanin_spill_start + spill_edge_count);
        }
    }
    reclaim_task_cursor = scan_end;
}

void PTO2FaninPool::ensure_space(
    PTO2SharedMemoryHandle &sm_handle, PTO2RingFlowControl &fc, uint8_t ring_id, int32_t needed
) {
    if (available() >= needed) return;

    int spin_count = 0;
    int32_t prev_last_alive = fc.last_task_alive.load(std::memory_order_acquire);
    while (available() < needed) {
        reclaim(sm_handle, ring_id, prev_last_alive);
        if (available() >= needed) return;

        spin_count++;

        int32_t cur_last_alive = fc.last_task_alive.load(std::memory_order_acquire);
        if (cur_last_alive > prev_last_alive) {
            spin_count = 0;
            prev_last_alive = cur_last_alive;
        }

        if (spin_count >= PTO2_DEP_POOL_SPIN_LIMIT) {
            int32_t current = fc.current_task_index.load(std::memory_order_acquire);
            LOG_ERROR("========================================");
            LOG_ERROR("FATAL: Fanin Spill Pool Deadlock Detected! (ring %d)", ring_id);
            LOG_ERROR("========================================");
            LOG_ERROR("Fanin spill pool cannot reclaim space after %d spins (no progress).", spin_count);
            LOG_ERROR(
                "  - Pool used:     %d / %d (%.1f%%)", used(), capacity,
                (capacity > 0) ? (100.0 * used() / capacity) : 0.0
            );
            LOG_ERROR("  - Pool top:      %d (linear)", top);
            LOG_ERROR("  - Pool tail:     %d (linear)", tail);
            LOG_ERROR("  - High water:    %d", high_water);
            LOG_ERROR("  - Needed:        %d entries", needed);
            LOG_ERROR("  - last_task_alive: %d (stuck here)", cur_last_alive);
            LOG_ERROR("  - current_task:    %d", current);
            LOG_ERROR("  - In-flight tasks: %d", current - cur_last_alive);
            LOG_ERROR("Diagnosis:");
            LOG_ERROR("  last_task_alive is not advancing, so fanin spill pool tail");
            LOG_ERROR("  cannot reclaim. Check TaskRing diagnostics for root cause.");
            LOG_ERROR("Solution:");
            LOG_ERROR("  Increase fanin spill pool capacity (current: %d, recommended: %d)", capacity, high_water * 2);
            LOG_ERROR("  Compile-time: PTO2_DEP_LIST_POOL_SIZE in pto_runtime2_types.h");
            LOG_ERROR("  Runtime env:  PTO2_RING_DEP_POOL=%d", high_water * 2);
            LOG_ERROR("========================================");
            exit(1);
        }
        SPIN_WAIT_HINT();
    }
}

// =============================================================================
// Dependency List Pool Implementation
// =============================================================================
void PTO2DepListPool::reclaim(PTO2SharedMemoryHandle &sm_handle, uint8_t ring_id, int32_t sm_last_task_alive) {
    if (sm_last_task_alive >= last_reclaimed + PTO2_DEP_POOL_CLEANUP_INTERVAL && sm_last_task_alive > 0) {
        int32_t mark = sm_handle.get_slot_state_by_task_id(ring_id, sm_last_task_alive - 1).dep_pool_mark;
        if (mark > 0) {
            advance_tail(mark);
        }
        last_reclaimed = sm_last_task_alive;
    }
}

void PTO2DepListPool::ensure_space(
    PTO2SharedMemoryHandle &sm_handle, PTO2RingFlowControl &fc, uint8_t ring_id, int32_t needed
) {
    if (available() >= needed) return;

    int spin_count = 0;
    int32_t prev_last_alive = fc.last_task_alive.load(std::memory_order_acquire);
    while (available() < needed) {
        reclaim(sm_handle, ring_id, prev_last_alive);
        if (available() >= needed) return;

        spin_count++;

        // Progress detection: reset spin counter if last_task_alive advances
        int32_t cur_last_alive = fc.last_task_alive.load(std::memory_order_acquire);
        if (cur_last_alive > prev_last_alive) {
            spin_count = 0;
            prev_last_alive = cur_last_alive;
        }

        if (spin_count >= PTO2_DEP_POOL_SPIN_LIMIT) {
            int32_t current = fc.current_task_index.load(std::memory_order_acquire);
            LOG_ERROR("========================================");
            LOG_ERROR("FATAL: Dependency Pool Deadlock Detected! (ring %d)", ring_id);
            LOG_ERROR("========================================");
            LOG_ERROR("DepListPool cannot reclaim space after %d spins (no progress).", spin_count);
            LOG_ERROR(
                "  - Pool used:     %d / %d (%.1f%%)", used(), capacity,
                (capacity > 0) ? (100.0 * used() / capacity) : 0.0
            );
            LOG_ERROR("  - Pool top:      %d (linear)", top);
            LOG_ERROR("  - Pool tail:     %d (linear)", tail);
            LOG_ERROR("  - High water:    %d", high_water);
            LOG_ERROR("  - Needed:        %d entries", needed);
            LOG_ERROR("  - last_task_alive: %d (stuck here)", cur_last_alive);
            LOG_ERROR("  - current_task:    %d", current);
            LOG_ERROR("  - In-flight tasks: %d", current - cur_last_alive);
            LOG_ERROR("Diagnosis:");
            LOG_ERROR("  last_task_alive is not advancing, so dep pool tail");
            LOG_ERROR("  cannot reclaim. Check TaskRing diagnostics for root cause.");
            LOG_ERROR("Solution:");
            LOG_ERROR("  Increase dep pool capacity (current: %d, recommended: %d)", capacity, high_water * 2);
            LOG_ERROR("  Compile-time: PTO2_DEP_LIST_POOL_SIZE in pto_runtime2_types.h");
            LOG_ERROR("  Runtime env:  PTO2_RING_DEP_POOL=%d", high_water * 2);
            LOG_ERROR("========================================");
            exit(1);
        }
        SPIN_WAIT_HINT();
    }
}
