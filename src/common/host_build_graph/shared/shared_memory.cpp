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
 * host_build_graph shared-memory implementation
 *
 * Implements shared memory allocation, initialization, and management
 * for Orchestrator-Scheduler communication.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "host_build_graph/shared_memory.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "host_build_graph/runtime_status.h"

// =============================================================================
// Size Calculation
// =============================================================================

uint64_t SharedMemoryHandle::calculate_size(uint64_t max_tasks) {
    // Total SM size = offset just past the last segment, from the single
    // source of truth for the layout (sm_layout::segment_offsets).
    return sm_layout::segment_offsets(max_tasks).end;
}

// =============================================================================
// Creation and Destruction
// =============================================================================

void SharedMemoryHandle::setup_pointers(uint64_t pitch) {
    char *base = (char *)sm_base;
    header = (SharedMemoryHeader *)base;

    // storage / task_states — offsets from the single source
    // of truth (sm_layout::segment_offsets), so this setup and the
    // device-address helpers cannot drift.
    //
    // Only the two slot-pitched segments, and no argument pool: the pool extents
    // differ between the mirror and the image, while these two depend on the pitch
    // alone. The orchestrator holds the mirror's pool bases; the device resolves an
    // argument region through the payload's delta and needs no base at all.
    auto off = sm_layout::segment_offsets(sm_layout::image_extents({pitch, 0, 0, 0}));
    auto &tasks = header->tasks;
    tasks.task_storage = (ChipTaskStorage *)(base + off.storage);
    tasks.task_states = (std::atomic<ChipTaskState> *)(base + off.task_states);
}

bool SharedMemoryHandle::init(void *sm_base_arg, uint64_t sm_size_arg, uint64_t max_tasks) {
    if (!sm_base_arg || sm_size_arg == 0) return false;
    const uint64_t mirror_bytes = calculate_size(max_tasks);
    // The mirror has to stay inside an int32 delta's reach: a payload names its
    // argument regions that way, and set() leaves a field unbound rather than
    // truncating, so a pool out of reach would read back as null and be dereferenced
    // as one. attach_populated makes the same check on the shipped image.
    //
    // Tested before the region's own size, because this rejects `max_tasks` itself:
    // no region, however large, makes such a reservation usable.
    if (mirror_bytes > static_cast<uint64_t>(INT32_MAX)) return false;
    if (sm_size_arg < mirror_bytes) return false;

    sm_base = sm_base_arg;
    sm_size = sm_size_arg;
    is_owner = false;
    setup_pointers(max_tasks);
    init_header();
    return true;
}

bool SharedMemoryHandle::attach_populated(
    void *sm_base_arg, uint64_t sm_size_arg, uint64_t max_tasks, uint64_t live_slots, uint64_t image_bytes
) {
    if (!sm_base_arg || sm_size_arg == 0) return false;
    // A pitch above `max_tasks` names a slot the host could not have built, and one
    // below what the host used would place every segment past the storage
    // short. This is the whole contract between the two sides, checked once at
    // attach rather than trusted.
    if (live_slots == 0 || live_slots > max_tasks) return false;
    // The image must at least hold the two slot-pitched segments; anything beyond
    // that is its argument pools, whose extents are the bind's cursors and are not
    // recomputable here. The first pool's offset is exactly where those two end.
    const uint64_t slots_end = sm_layout::segment_offsets(sm_layout::image_extents({live_slots, 0, 0, 0})).fanin_pool;
    if (image_bytes < slots_end) return false;
    // The region holds the compacted image, so that is what has to fit — the
    // dimensioned size is no longer its size.
    if (sm_size_arg < image_bytes) return false;
    // A payload names its argument regions by an int32 delta from its own address, so
    // no two positions in the image may be further apart than that.
    if (image_bytes > static_cast<uint64_t>(INT32_MAX)) return false;

    sm_base = sm_base_arg;
    sm_size = sm_size_arg;
    is_owner = false;
    setup_pointers(live_slots);
    // Deliberately NO init_header: the SM already holds the host orchestrator's
    // task graph (storage entries and task states).
    return true;
}

SharedMemoryHandle *SharedMemoryHandle::create_and_init_default(DeviceArena &arena) {
    const uint64_t buffer_size = calculate_size(CHIP_DEFAULT_GRAPH_TASKS);
    const size_t off_handle = arena.reserve(sizeof(SharedMemoryHandle), alignof(SharedMemoryHandle));
    const size_t off_buffer = arena.reserve(static_cast<size_t>(buffer_size), CHIP_ALIGN_SIZE);
    if (arena.commit() == nullptr) return nullptr;

    auto *handle = static_cast<SharedMemoryHandle *>(arena.region_ptr(off_handle));
    memset(handle, 0, sizeof(*handle));
    void *buffer = arena.region_ptr(off_buffer);
    memset(buffer, 0, static_cast<size_t>(buffer_size));
    if (!handle->init(buffer, buffer_size, CHIP_DEFAULT_GRAPH_TASKS)) return nullptr;
    return handle;
}

void SharedMemoryHandle::destroy() {
    // Arena-owned wrappers (is_owner == false) are reclaimed by arena.release();
    // calling destroy on them is a no-op so existing callers stay safe.
    if (is_owner && sm_base) {
        free(sm_base);
        free(this);
    }
}

// =============================================================================
// Initialization
// =============================================================================
//
// no need init data in pool, init pool data when used
void SharedMemoryHandle::init_header() {
    // 0 until run_host_orchestration writes the run's count; init runs before
    // orchestration, so the real value is not known here yet.
    header->tasks.total_tasks = 0;

    header->reset_execution_state();

    // Per-slot init (slot.reset_for_reuse() + active_mask, and clearing the
    // progress state) happens init-on-write in orch::prepare_task as each slot
    // [0, total_tasks) is claimed, so the SM init/upload cost tracks the task
    // count, not the size the table was dimensioned for. The device reads no slot
    // past total_tasks, so the unclaimed tail is left uninitialized.
}
