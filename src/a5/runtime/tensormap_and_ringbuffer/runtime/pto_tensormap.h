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
 * PTO Runtime2 - TensorMap Interface
 *
 * TensorMap provides producer lookup for dependency discovery:
 * - Maps Tensor -> producer task ID
 * - Used by pto_submit_task() to find dependencies
 *
 * Key design features:
 * 1. Ring buffer pool for entries (no malloc/free)
 * 2. Lazy invalidation (entries become stale when producer retires)
 * 3. Per-task per-ring entry tracking for efficient cleanup
 * 4. OVERLAP DETECTION: Detects dependencies for overlapping sub-regions
 *
 * Hash table with chaining:
 * - buckets[] array of head offsets
 * - Entries linked via next_in_bucket
 * - Insert at head (newest first) for sorted chains
 *
 * CRITICAL: Hash only by base_ptr
 * ==============================
 * For overlap detection to work, ALL sub-regions of the same base tensor
 * MUST be in the SAME hash bucket. This allows lookup to compare all
 * potentially overlapping regions.
 *
 * Overlap detection: Two regions create a dependency if:
 *   1. Same base_ptr (raw tensor pointer)
 *   2. Byte ranges [offset, offset+size) intersect
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include "common.h"              // NOLINT(build/include_subdir)
#include "pto_runtime2_types.h"  // NOLINT(build/include_subdir)
#include "tensor.h"              // NOLINT(build/include_subdir)

struct PTO2OrchestratorState;  // forward declare

// =============================================================================
// TensorMap Lookup Profiling (must precede inline lookup/insert methods)
// =============================================================================
#ifndef PTO2_TENSORMAP_PROFILING
#define PTO2_TENSORMAP_PROFILING 0
#endif

#if PTO2_TENSORMAP_PROFILING
extern uint64_t g_lookup_chain_total;
extern uint64_t g_lookup_count;
extern int32_t g_lookup_chain_max;
extern uint64_t g_lookup_overlap_checks;
extern uint64_t g_lookup_overlap_hits;
extern uint64_t g_insert_count;
#endif

// =============================================================================
// TensorMap Structure
// =============================================================================

/**
 * TensorMap entry structure — cache-line optimized for lookup
 *
 * Cache line 1 (64B, lookup hot path):
 *   next_in_bucket, producer_task_id, buffer_addr — chain traversal + validity + hash match
 *   version, ndims, is_all_offset_zero, bucket_index — overlap fast path
 *   shapes[5] — overlap comparison
 *
 * Cache line 2 (64B, insert/remove/slow-path only):
 *   prev_in_bucket, next_in_task, prev_in_task — chain manipulation
 *   offsets[5] — only read when !is_all_offset_zero
 *
 * When is_all_offset_zero is true, lookup touches only cache line 1.
 * Entry size: 128B (2 cache lines) vs previous 192B (3 cache lines with embedded Tensor).
 */
struct alignas(64) PTO2TensorMapEntry {
    // === Cache line 1 (64B) — lookup hot path ===
    uint64_t buffer_addr;                // 8B: tensor base address (hash key)
    PTO2TensorMapEntry *next_in_bucket;  // 8B: next entry in hash bucket chain
    PTO2TaskId producer_task_id;         // 8B: raw (ring_id << 32) | local_id
    int32_t bucket_index;                // 4B: bucket index (-1 if unlinked)
    uint32_t __padding0__;               // 4B: occupies Tensor::start_offset high half
    int32_t version;                     // 4B: tensor version for overlap detection
    uint32_t ndims;                      // 4B: number of dimensions
    DataType __padding_dtype__;          // 1B: occupies Tensor::dtype
    bool is_all_offset_zero;             // 1B: fast-path flag
    uint8_t __padding1__[2];
    uint32_t shapes[RUNTIME_MAX_TENSOR_DIMS];  // 20B: shape per dimension

    // === Cache line 2 (64B) — insert/remove/slow-path ===
    PTO2TensorMapEntry *prev_in_bucket;         // 8B: prev in hash bucket chain
    PTO2TensorMapEntry *next_in_task;           // 8B: next entry for same task
    PTO2TensorMapEntry *prev_in_task;           // 8B: prev entry for same task
    uint32_t offsets[RUNTIME_MAX_TENSOR_DIMS];  // 20B: only when !is_all_offset_zero
    // padding: 20B to fill 64B

    /**
     * Copy overlap-relevant fields from a Tensor into this entry.
     */
    void copy_from_tensor(const Tensor &tensor) {
        memcpy(this, &tensor, 64);
        if (!tensor.is_all_offset_zero) {
            for (uint32_t i = 0; i < tensor.ndims; i++) {
                offsets[i] = tensor.offsets[i];
            }
        }
    }

    void copy_tensor_create_info(const TensorCreateInfo &tensor_create_info, uint64_t addr) {
        memcpy(this, &tensor_create_info, 64);
        buffer_addr = addr;
    }

    /**
     * Check overlap between input tensor and this entry (the producer output).
     * Mirrors Tensor::is_overlap() logic but operates on entry fields directly.
     */
    OverlapStatus check_overlap(const Tensor &input) const {
        debug_assert(input.buffer.addr == buffer_addr);
        debug_assert(input.version >= version);
        if (input.version > version) {
            return OverlapStatus::OTHER;
        }
        // Fast path: both have zero offsets → ranges are [0, shape[i])
        if (input.is_all_offset_zero && is_all_offset_zero) {
            bool contains = true;
            for (uint32_t i = 0; i < ndims; i++) {
                if (input.shapes[i] < shapes[i]) {
                    contains = false;
                    break;
                }
            }
            return contains ? OverlapStatus::COVERED : OverlapStatus::OTHER;
        }
        // Slow path: at least one has non-zero offsets
        bool contains = true;
        for (uint32_t i = 0; i < ndims; i++) {
            uint64_t in_off = input.is_all_offset_zero ? 0 : input.offsets[i];
            uint64_t ent_off = is_all_offset_zero ? 0 : offsets[i];
            Segment in_range{in_off, in_off + static_cast<uint64_t>(input.shapes[i])};
            Segment ent_range{ent_off, ent_off + static_cast<uint64_t>(shapes[i])};
            if (!in_range.line_segment_intersection(ent_range)) {
                return OverlapStatus::NO_OVERLAP;
            } else if (!in_range.contains(ent_range)) {
                contains = false;
            }
        }
        return contains ? OverlapStatus::COVERED : OverlapStatus::OTHER;
    }
};

static_assert(sizeof(PTO2TensorMapEntry) == 128, "TensorMapEntry must be exactly 2 cache lines (128 bytes)");
static_assert(offsetof(PTO2TensorMapEntry, buffer_addr) == offsetof(Tensor, buffer.addr));
static_assert(offsetof(PTO2TensorMapEntry, version) == offsetof(Tensor, version));
static_assert(offsetof(PTO2TensorMapEntry, ndims) == offsetof(Tensor, ndims));
static_assert(offsetof(PTO2TensorMapEntry, is_all_offset_zero) == offsetof(Tensor, is_all_offset_zero));
static_assert(offsetof(PTO2TensorMapEntry, shapes) == offsetof(Tensor, shapes));
static_assert(
    offsetof(PTO2TensorMapEntry, prev_in_bucket) == 64, "TensorMapEntry must be exactly 2 cache lines (128 bytes)"
);

/**
 * Stack-allocated lookup result (avoids heap allocation per lookup)
 */
#define PTO2_LOOKUP_MAX_RESULTS 16
// =============================================================================
// TensorMap Lookup Chain Length Statistics (compile-time toggle)
// =============================================================================
struct PTO2LookupResult {
    struct Entry {
        PTO2TensorMapEntry *entry;
        OverlapStatus overlap_status;
    };
    Entry entries[PTO2_LOOKUP_MAX_RESULTS];
    int32_t count{0};

    void push(PTO2TensorMapEntry *entry, OverlapStatus s) {
        if (count < PTO2_LOOKUP_MAX_RESULTS) {
            entries[count++] = {entry, s};
        }
    }
};

/**
 * TensorMap structure
 *
 * Hash table with ring buffer entry pool and lazy invalidation.
 */
struct PTO2TensorMap {
    // Hash table buckets (fixed size, power of 2)
    PTO2TensorMapEntry **buckets;  // Array of offsets into entry_pool (-1 = empty)
    int32_t num_buckets;           // Must be power of 2 for fast modulo

    // Entry pool as ring buffer
    PTO2TensorMapEntry *entry_pool;        // Ring buffer of entries
    PTO2TensorMapEntry **free_entry_list;  // free entry ids
    int32_t pool_size;                     // Total pool capacity
    int32_t next_entry_idx;                // id when next entry insert
    int32_t free_num;                      // free entry number in entry pool

    // Per-ring per-task entry tracking (for efficient bucket cleanup)
    // Indexed by [ring_id][local_id & (task_window_sizes[ring_id] - 1)]
    PTO2TensorMapEntry **task_entry_heads[PTO2_MAX_RING_DEPTH];
    int32_t task_window_sizes[PTO2_MAX_RING_DEPTH];  // Per-ring task window size (for slot masking)

    // Per-ring validity threshold (for lazy invalidation)
    int32_t last_task_alives[PTO2_MAX_RING_DEPTH];  // Cached from shared memory per ring

    // Per-ring cleanup progress (for periodic cleanup_retired)
    int32_t last_cleanup[PTO2_MAX_RING_DEPTH]{};

    PTO2OrchestratorState *orch{nullptr};

    // new_entry only allocates memory, does not assign attributes
    PTO2TensorMapEntry *new_entry() {
        if (free_num > 0) {
            PTO2TensorMapEntry *res = free_entry_list[--free_num];
            debug_assert(res->bucket_index == -1);
            return res;
        }
        always_assert(next_entry_idx < pool_size);
        PTO2TensorMapEntry *res = &entry_pool[next_entry_idx++];
        debug_assert(res->bucket_index == -1);
        return res;
    }

    void free_entry(PTO2TensorMapEntry &entry) {
        always_assert(entry.bucket_index != -1);  // must still be in a bucket

        // Update predecessor's next pointer (O(1) via prev_in_bucket)
        if (entry.prev_in_bucket == nullptr) {
            // Entry is the head of its bucket chain, update bucket head
            // Must compute hash BEFORE clearing tensor
            buckets[entry.bucket_index] = entry.next_in_bucket;
        } else {
            entry.prev_in_bucket->next_in_bucket = entry.next_in_bucket;
        }

        // Update successor's prev pointer
        if (entry.next_in_bucket != nullptr) {
            entry.next_in_bucket->prev_in_bucket = entry.prev_in_bucket;
        }

        free_entry_list[free_num++] = &entry;
        entry.bucket_index = -1;
        entry.next_in_bucket = nullptr;
        entry.prev_in_bucket = nullptr;
        entry.next_in_task = nullptr;
        entry.prev_in_task = nullptr;
    }

    // =============================================================================
    // TensorMap API
    // =============================================================================

    /**
     * Initialize TensorMap
     *
     * @param num_buckets Number of hash buckets (must be power of 2)
     * @param pool_size   Size of entry pool
     * @return true on success, false on allocation failure
     */
    bool init(int32_t num_buckets, int32_t pool_size, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH]);

    /**
     * Initialize TensorMap with default sizes
     */
    bool init_default(const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH]);

    /**
     * Destroy TensorMap and free resources
     */
    void destroy();

    /**
     * Update validity threshold from shared memory
     * Called periodically to refresh the lazy invalidation threshold.
     *
     * @param last_task_alive  Current value from shared memory
     */
    void sync_validity(int32_t ring_id, int32_t last_task_alive) { this->last_task_alives[ring_id] = last_task_alive; }

    /**
     * Lookup producer for a tensor region
     *
     * Searches the hash table for a matching region.
     * Returns producer entry if found and valid.
     * Stale entries from different rings are skipped (not truncated).
     *
     * @param tensor  Tensor to look up
     * @param result  Output: stack-allocated result buffer
     */
    void lookup(const Tensor &tensor, PTO2LookupResult &result) {
        uint32_t bucket_index = hash(tensor.buffer.addr);
        PTO2TensorMapEntry *cur_entry = buckets[bucket_index];

        result.count = 0;
#if PTO2_TENSORMAP_PROFILING
        g_lookup_count++;
        int32_t chain_len = 0;
#endif

        while (cur_entry != nullptr) {
            PTO2TensorMapEntry *next_entry = cur_entry->next_in_bucket;

#if PTO2_TENSORMAP_PROFILING
            chain_len++;
#endif
            // Skip stale entries (no chain truncation — entries from different
            // rings can be interleaved, so a stale entry from one ring does NOT
            // imply subsequent entries from other rings are also stale)
            if (!entry_valid(*cur_entry)) {
                cur_entry = next_entry;
                continue;
            }

            // Entry is valid - check if regions OVERLAP (not just exact match)
            // Since we hash only by base_ptr, all entries in this bucket have
            // potential to overlap. We must check actual byte-range overlap.
            if (tensor.buffer.addr == cur_entry->buffer_addr) {
#if PTO2_TENSORMAP_PROFILING
                g_lookup_overlap_checks++;
#endif
                auto overlap_status = cur_entry->check_overlap(tensor);
                if (overlap_status != OverlapStatus::NO_OVERLAP) {
                    result.push(cur_entry, overlap_status);
#if PTO2_TENSORMAP_PROFILING
                    g_lookup_overlap_hits++;
#endif
                }
            }

            // Move to next entry
            cur_entry = next_entry;
        }
#if PTO2_TENSORMAP_PROFILING
        g_lookup_chain_total += chain_len;
        if (chain_len > g_lookup_chain_max) g_lookup_chain_max = chain_len;
#endif
    }

    /**
     * Insert a new entry (called when task produces output)
     *
     * Allocates from ring buffer pool, may overwrite stale entries.
     * Inserts at head of hash bucket chain (maintains task_id ordering).
     *
     * @param tensor            Tensor produced
     * @param producer_task_id  Task ID of producer
     */
    void insert(const Tensor &tensor, PTO2TaskId producer_task_id) {
        PTO2TensorMapEntry *entry = new_entry();
        entry->copy_from_tensor(tensor);
        link_entry(entry, tensor.buffer.addr, producer_task_id);
    }

    /**
     * Cleanup stale entries for retired tasks
     *
     * Called periodically by Orchestrator when last_task_alive advances.
     * Removes entries from bucket chains for tasks in [old, new) range.
     *
     * @param old_last_task_alive  Previous threshold
     * @param new_last_task_alive  New threshold
     */
    void cleanup_retired(int32_t ring_id, int32_t old_last_task_alive, int32_t new_last_task_alive) {
        // Iterate through retired tasks on this ring and remove their entries
        for (int32_t local_id = old_last_task_alive; local_id < new_last_task_alive; local_id++) {
            int32_t task_slot = local_id & (task_window_sizes[ring_id] - 1);
            PTO2TensorMapEntry *cur_entry = task_entry_heads[ring_id][task_slot];

            while (cur_entry != nullptr) {
                PTO2TensorMapEntry *next_entry = cur_entry->next_in_task;  // Save before clearing
                // Only remove if this entry belongs to the retiring task
                // (slot may have been reused by a newer task)
                debug_assert(
                    cur_entry->producer_task_id ==
                    PTO2TaskId::make(static_cast<uint8_t>(ring_id), static_cast<uint32_t>(local_id))
                );
                free_entry(*cur_entry);
                cur_entry = next_entry;
            }

            // Clear task's entry head (slot will be reused by local_id + task_window_sizes[ring_id])
            task_entry_heads[ring_id][task_slot] = nullptr;
        }
    }

    // =============================================================================
    // Internal Helpers (exposed for testing)
    // =============================================================================

    /**
     * Compute hash for tensor addr
     *
     * Multiplicative hash using the golden-ratio constant.  Multiplication
     * mixes ALL input bits into the high bits of the product, so aligned
     * addresses (low bits all-zero) still distribute evenly.  We extract
     * the top log2(num_buckets) bits which carry the most entropy.
     */
    uint32_t hash(uint64_t key) {
        key *= 0x9E3779B97F4A7C15ULL;
        return static_cast<uint32_t>(key >> (64 - __builtin_ctz(num_buckets)));
    }

    /**
     * Link an initialized entry into bucket and task chains.
     */
    void link_entry(PTO2TensorMapEntry *entry, uint64_t addr, PTO2TaskId producer_task_id) {
#if PTO2_TENSORMAP_PROFILING
        g_insert_count++;
#endif
        uint32_t bucket_index = hash(addr);
        auto ring_id = producer_task_id.ring();
        auto local_id = producer_task_id.local();
        int32_t task_slot = local_id & (task_window_sizes[ring_id] - 1);

        entry->producer_task_id = producer_task_id;

        // Insert at head of hash bucket
        entry->bucket_index = bucket_index;
        entry->next_in_bucket = buckets[bucket_index];
        if (entry->next_in_bucket != nullptr) {
            entry->next_in_bucket->prev_in_bucket = entry;
        }
        buckets[bucket_index] = entry;
        entry->prev_in_bucket = nullptr;

        // Link to task's entry list
        entry->next_in_task = task_entry_heads[ring_id][task_slot];
        entry->prev_in_task = nullptr;
        if (entry->next_in_task != nullptr) {
            entry->next_in_task->prev_in_task = entry;
        }
        task_entry_heads[ring_id][task_slot] = entry;
    }

    /**
     * Check if entry is valid (producer has not retired)
     */
    bool entry_valid(const PTO2TensorMapEntry &entry) const {
        return static_cast<int32_t>(entry.producer_task_id.local()) >= last_task_alives[entry.producer_task_id.ring()];
    }

    void remove_entry(PTO2TensorMapEntry &entry) {
        remove_from_task(entry);
        free_entry(entry);
    }

    /**
     * Remove entry from its task chain (O(1) with prev pointer)
     * Called during pool wrap-around to unlink reused entries.
     */
    void remove_from_task(PTO2TensorMapEntry &entry) {
        always_assert(entry.bucket_index != -1);  // must still be in a bucket
        // Update predecessor's next pointer (O(1) via prev_in_task)
        if (entry.prev_in_task == nullptr) {
            // Entry is the head of its task chain, update task_entry_heads
            int32_t ring_id = entry.producer_task_id.ring();
            int32_t local_id = static_cast<int32_t>(entry.producer_task_id.local());
            int32_t task_slot = local_id & (task_window_sizes[ring_id] - 1);
            task_entry_heads[ring_id][task_slot] = entry.next_in_task;
        } else {
            entry.prev_in_task->next_in_task = entry.next_in_task;
        }

        // Update successor's prev pointer
        if (entry.next_in_task != nullptr) {
            entry.next_in_task->prev_in_task = entry.prev_in_task;
        }

        entry.next_in_task = nullptr;
        entry.prev_in_task = nullptr;
    }

    // =============================================================================
    // Debug Utilities
    // =============================================================================

    /**
     * Print TensorMap statistics
     */
    void print_stats();

    /**
     * Get count of valid entries
     */
    int32_t valid_count();

    // =============================================================================
    // TensorMap Synchronization
    // =============================================================================

    /**
     * Sync TensorMap validity threshold from shared memory
     *
     * Called periodically to refresh the lazy invalidation threshold.
     * Also triggers cleanup if threshold has advanced significantly.
     */
    void sync_tensormap(uint8_t ring_id, int32_t sm_last_task_alive);
};

#if PTO2_TENSORMAP_PROFILING
struct PTO2TensorMapProfilingData {
    uint64_t lookup_chain_total;
    uint64_t lookup_count;
    int32_t lookup_chain_max;
    uint64_t overlap_checks;
    uint64_t overlap_hits;
    uint64_t insert_count;
};

PTO2TensorMapProfilingData pto2_tensormap_get_profiling();
#endif
