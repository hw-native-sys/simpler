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

#include "pto_tensormap.h"

#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "common/unified_log.h"

PTO2TensorMapLayout PTO2TensorMap::reserve_layout(
    DeviceArena &arena, int32_t new_num_buckets, int32_t new_pool_size,
    const int32_t new_task_window_sizes[PTO2_MAX_RING_DEPTH]
) {
    // num_buckets must be a positive power of two for the hash truncation to
    // work (0 passes the power-of-two test but makes __builtin_ctz UB).
    always_assert(new_num_buckets > 0 && (new_num_buckets & (new_num_buckets - 1)) == 0);

    PTO2TensorMapLayout layout{};
    layout.num_buckets = new_num_buckets;
    layout.pool_size = new_pool_size;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        layout.task_window_sizes[r] = new_task_window_sizes[r];

    layout.off_buckets = arena.reserve(
        static_cast<size_t>(new_num_buckets) * sizeof(PTO2TensorMapEntry *), alignof(PTO2TensorMapEntry *)
    );
    layout.off_bucket_epochs =
        arena.reserve(static_cast<size_t>(new_num_buckets) * sizeof(uint32_t), alignof(uint32_t));
    layout.off_entry_pool =
        arena.reserve(static_cast<size_t>(new_pool_size) * sizeof(PTO2TensorMapEntry), alignof(PTO2TensorMapEntry));
    layout.off_free_entry_list =
        arena.reserve(static_cast<size_t>(new_pool_size) * sizeof(PTO2TensorMapEntry *), alignof(PTO2TensorMapEntry *));
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        layout.off_task_entry_heads[r] = arena.reserve(
            static_cast<size_t>(new_task_window_sizes[r]) * sizeof(PTO2TensorMapEntry *), alignof(PTO2TensorMapEntry *)
        );
        layout.off_task_entry_head_epochs[r] =
            arena.reserve(static_cast<size_t>(new_task_window_sizes[r]) * sizeof(uint32_t), alignof(uint32_t));
    }
    return layout;
}

PTO2TensorMapLayout
PTO2TensorMap::reserve_layout_default(DeviceArena &arena, const int32_t new_task_window_sizes[PTO2_MAX_RING_DEPTH]) {
    return reserve_layout(arena, PTO2_TENSORMAP_NUM_BUCKETS, PTO2_TENSORMAP_POOL_SIZE, new_task_window_sizes);
}

bool PTO2TensorMap::init_data_from_layout(const PTO2TensorMapLayout &layout, DeviceArena &arena) {
    num_buckets = layout.num_buckets;
    pool_size = layout.pool_size;

    // Address arena regions for data writes; do not store these in struct
    // fields (wire_arena_pointers does that).
    auto *buckets_arena = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_buckets));
    auto *bucket_epochs_arena = static_cast<uint32_t *>(arena.region_ptr(layout.off_bucket_epochs));
    auto *entry_pool_arena = static_cast<PTO2TensorMapEntry *>(arena.region_ptr(layout.off_entry_pool));
    auto *free_list_arena = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_free_entry_list));

    // buckets[]: empty == nullptr.
    for (int32_t i = 0; i < num_buckets; i++) {
        buckets_arena[i] = nullptr;
        bucket_epochs_arena[i] = 0;
    }

    memset(entry_pool_arena, 0, static_cast<size_t>(pool_size) * sizeof(PTO2TensorMapEntry));
    for (int32_t i = 0; i < pool_size; i++) {
        entry_pool_arena[i].bucket_index = -1;
        entry_pool_arena[i].next_in_bucket = nullptr;
        entry_pool_arena[i].prev_in_bucket = nullptr;
        entry_pool_arena[i].next_in_task = nullptr;
        entry_pool_arena[i].prev_in_task = nullptr;
        entry_pool_arena[i].producer_task_id = PTO2TaskId{};
    }

    // free_entry_list: zeroed (was calloc'd before); contents become meaningful
    // only after entries are freed back, so the body of the array stays as 0.
    memset(free_list_arena, 0, static_cast<size_t>(pool_size) * sizeof(PTO2TensorMapEntry *));

    next_entry_idx = 0;
    free_num = 0;

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        auto *heads_arena = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_task_entry_heads[r]));
        auto *head_epochs_arena = static_cast<uint32_t *>(arena.region_ptr(layout.off_task_entry_head_epochs[r]));
        for (int32_t i = 0; i < layout.task_window_sizes[r]; i++) {
            heads_arena[i] = nullptr;
            head_epochs_arena[i] = 0;
        }
        task_window_sizes[r] = layout.task_window_sizes[r];
        last_task_alives[r] = 0;
        last_cleanup[r] = 0;
    }

    return true;
}

void PTO2TensorMap::wire_arena_pointers(const PTO2TensorMapLayout &layout, DeviceArena &arena) {
    buckets = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_buckets));
    bucket_epochs = static_cast<uint32_t *>(arena.region_ptr(layout.off_bucket_epochs));
    entry_pool = static_cast<PTO2TensorMapEntry *>(arena.region_ptr(layout.off_entry_pool));
    free_entry_list = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_free_entry_list));
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        task_entry_heads[r] = static_cast<PTO2TensorMapEntry **>(arena.region_ptr(layout.off_task_entry_heads[r]));
        task_entry_head_epochs[r] = static_cast<uint32_t *>(arena.region_ptr(layout.off_task_entry_head_epochs[r]));
    }
}

void PTO2TensorMap::reset_for_reuse() {
    next_entry_idx = 0;
    free_num = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        last_task_alives[r] = 0;
        last_cleanup[r] = 0;
    }
    current_epoch++;
    if (current_epoch == 0) {
        current_epoch = 1;
        for (int32_t i = 0; i < num_buckets; i++)
            bucket_epochs[i] = 0;
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            for (int32_t i = 0; i < task_window_sizes[r]; i++)
                task_entry_head_epochs[r][i] = 0;
        }
    }
}

void PTO2TensorMap::destroy() {
    buckets = nullptr;
    entry_pool = nullptr;
    free_entry_list = nullptr;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        task_entry_heads[r] = nullptr;
}

int32_t PTO2TensorMap::valid_count() {
    int32_t count = 0;

    for (int32_t i = 0; i < pool_size; i++)
        if (entry_pool[i].bucket_index != -1 && entry_valid(entry_pool[i])) count++;

    return count;
}

void PTO2TensorMap::sync_tensormap(PTO2TaskId task_id, int32_t sm_last_task_alive) {
    auto ring_id = task_id.ring();
    auto local_id = task_id.local();
    sync_validity(ring_id, sm_last_task_alive);

    // Only attempt cleanup when last_task_alive has actually advanced;
    // otherwise cleanup_retired would empty-loop and we'd spin forever.
    auto overlap = get_task_local_id_slot(ring_id, local_id) == get_task_local_id_slot(ring_id, last_cleanup[ring_id]);
    if (sm_last_task_alive - last_cleanup[ring_id] >= PTO2_TENSORMAP_CLEANUP_INTERVAL || overlap) {
        cleanup_retired(ring_id, last_cleanup[ring_id], sm_last_task_alive);
        last_cleanup[ring_id] = sm_last_task_alive;
    }
}
