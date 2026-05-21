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

#include "common.h"
#include "device_arena.h"
#include "pto_runtime2_types.h"
#include "tensor.h"

struct PTO2OrchestratorState;  // forward declare

/**
 * Layout descriptor produced by PTO2TensorMap::reserve_layout(). Stores the
 * region offsets returned by DeviceArena::reserve() so init_from_layout()
 * can fetch the matching pointers after the arena is committed.
 *
 * All offsets are relative to the arena's base.
 */
struct PTO2TensorMapLayout {
    size_t off_buckets;
    size_t off_entry_pool;
    size_t off_free_entry_list;
    size_t off_task_entry_heads[PTO2_MAX_RING_DEPTH];
    int32_t num_buckets;
    int32_t pool_size;
    int32_t task_window_sizes[PTO2_MAX_RING_DEPTH];
};

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
 * Cache line 1 (64B, lookup hot path) mirrors Tensor cache line 1 byte-for-byte
 * from byte 16 onward, so that `memcpy(this, &tensor, 64)` populates everything
 * we need for overlap checks. Bytes [0, 16) carry entry-only fields (hash
 * bucket head + chain pointer) that overlap Tensor::buffer (addr in [0, 8) is
 * the hash key, size in [8, 16) is unused by the entry — we repurpose it for
 * `next_in_bucket`).
 *
 *   buffer_addr / next_in_bucket / producer_task_id   — chain traversal + match
 *   start_offset                                       — overlap byte range begin
 *   version, ndims, dtype, manual_dep, is_contiguous   — overlap fast path
 *   shapes[5]                                          — overlap comparison (line 1)
 *
 * Cache line 2 (64B, slow-path / non-contiguous overlap):
 *   prev_in_bucket / next_in_task / prev_in_task       — chain manipulation
 *   bucket_index                                       — bookkeeping
 *   extent_elem_cache                                  — overlap byte range end
 *   strides[5]                                          — reserved for L2 overlap (PR-2)
 *
 * When both entry & probe are `is_contiguous && start_offset == 0`, the overlap
 * check derives `extent_elem = prod(shapes)` from cache line 1 alone.
 *
 * Entry size: 128B (2 cache lines), matches Tensor.
 */
struct alignas(64) PTO2TensorMapEntry {
    // === Cache line 1 (64B) — lookup hot path; mirrors Tensor line 1 from byte 16 ===
    uint64_t buffer_addr;                // 8B [0, 8):   tensor base address (hash key, mirrors Tensor::buffer.addr)
    PTO2TensorMapEntry *next_in_bucket;  // 8B [8, 16):  next entry in hash bucket chain (overlays Tensor::buffer.size)
    PTO2TaskId producer_task_id;         // 8B [16,24):  mirrors Tensor::owner_task_id slot
    uint64_t start_offset;               // 8B [24,32):  mirrors Tensor::start_offset (element offset)
    int32_t version;                     // 4B [32,36):  mirrors Tensor::version
    uint32_t ndims;                      // 4B [36,40):  mirrors Tensor::ndims
    DataType dtype;                      // 1B [40,41):  mirrors Tensor::dtype
    bool manual_dep;                     // 1B [41,42):  mirrors Tensor::manual_dep
    bool is_contiguous;                  // 1B [42,43):  mirrors Tensor::is_contiguous
    uint8_t __padding1__;                // 1B [43,44):  mirrors Tensor padding
    uint32_t shapes[RUNTIME_MAX_TENSOR_DIMS];  // 20B [44,64): mirrors Tensor::shapes

    // === Cache line 2 (64B) — chain manipulation + non-contiguous overlap data ===
    PTO2TensorMapEntry *prev_in_bucket;         // 8B [64, 72)
    PTO2TensorMapEntry *next_in_task;           // 8B [72, 80)
    PTO2TensorMapEntry *prev_in_task;           // 8B [80, 88)
    int32_t bucket_index;                       // 4B [88, 92): -1 when unlinked
    uint32_t __padding2__;                      // 4B [92, 96)
    uint64_t extent_elem_cache;                 // 8B [96,104): non-contiguous extent (mirrors Tensor)
    uint32_t strides[RUNTIME_MAX_TENSOR_DIMS];  // 20B [104,124): element strides, mirrors Tensor::strides
    uint8_t __padding3__[4];                    // 4B [124,128)

    /**
     * Copy overlap-relevant fields from a Tensor into this entry.
     *
     * 64B memcpy of Tensor cache line 1 populates buffer_addr (byte [0,8)),
     * producer_task_id, start_offset, version, ndims, dtype, manual_dep,
     * is_contiguous and shapes[]. Byte [8,16) holds Tensor::buffer.size in
     * the source and gets written into next_in_bucket; that's harmless
     * because link_entry() overwrites next_in_bucket immediately after.
     *
     * Cache line 2 (stride / extent_elem_cache) is derived from line 1 when
     * the source is canonically contiguous (is_contiguous && start_offset==0),
     * so the producer Tensor's cache line 2 stays cold during insert. Only
     * non-contiguous producers pay one extra line 2 read.
     */
    void copy_from_tensor(const Tensor &tensor) {
        memcpy(this, &tensor, 64);
        if (tensor.is_contiguous && tensor.start_offset == 0) {
            uint64_t numel = 1;
            for (uint32_t i = 0; i < tensor.ndims; i++)
                numel *= tensor.shapes[i];
            extent_elem_cache = numel;
            uint32_t s = 1;
            for (int32_t i = static_cast<int32_t>(tensor.ndims) - 1; i >= 0; i--) {
                strides[i] = s;
                s *= tensor.shapes[i];
            }
        } else {
            extent_elem_cache = tensor.extent_elem_cache;
            for (uint32_t i = 0; i < tensor.ndims; i++) {
                strides[i] = tensor.strides[i];
            }
        }
    }

    void copy_tensor_create_info(const TensorCreateInfo &tensor_create_info, uint64_t addr) {
        memcpy(this, &tensor_create_info, 64);
        buffer_addr = addr;
        // Create-info outputs are always contiguous with start_offset = 0;
        // extent_elem = prod(shapes); stride is row-major.
        uint64_t numel = 1;
        for (uint32_t i = 0; i < tensor_create_info.ndims; i++) {
            numel *= tensor_create_info.shapes[i];
        }
        extent_elem_cache = numel;
        uint32_t s = 1;
        for (int32_t i = static_cast<int32_t>(tensor_create_info.ndims) - 1; i >= 0; i--) {
            strides[i] = s;
            s *= tensor_create_info.shapes[i];
        }
    }

    /**
     * Effective element extent of this entry.
     * Contiguous-aligned views compute it from shapes alone (line 1 hit only);
     * non-contiguous views read the cached value from line 2.
     */
    uint64_t effective_extent_elem() const {
        if (is_contiguous) {
            uint64_t n = 1;
            for (uint32_t i = 0; i < ndims; i++)
                n *= shapes[i];
            return n;
        }
        return extent_elem_cache;
    }

    /**
     * Check overlap between input tensor and this entry (the producer output).
     *
     * Three-level cascade:
     *   L1 — O(1) byte-range intersection. Disjoint -> NO_OVERLAP.
     *   L2 — O(ndims) hyper-rectangle precise check, eligible only when both
     *        sides share the same canonical row-major axis layout (same
     *        dtype/ndims/strides[], stride descends as integer multiples,
     *        start_offset decomposes cleanly under the reference shape).
     *        Yields NO_OVERLAP / COVERED / OTHER per-dim.
     *   L3 — Non-hyper-rectangle pairs (transpose/permute mismatch, slice
     *        with step, etc): conservative OTHER. Exact enumeration via
     *        contiguous-segment merge is scheduled for a follow-up.
     *
     * COVERED is returned when `input` completely contains `entry` per-dim
     * — dep_compute uses this to retire the now-redundant entry.
     */
    OverlapStatus check_overlap(const Tensor &input) const {
        debug_assert(input.buffer.addr == buffer_addr);
        debug_assert(input.version >= version);
        if (input.version > version) {
            return OverlapStatus::OTHER;
        }

        // -------- L1: byte-range intersection (O(1) fast reject) --------
        const uint64_t in_begin = input.start_offset;
        const uint64_t in_end = input.start_offset + input.extent_elem();
        const uint64_t ent_begin = start_offset;
        const uint64_t ent_end = start_offset + effective_extent_elem();
        Segment in_range_bytes{in_begin, in_end};
        Segment ent_range_bytes{ent_begin, ent_end};
        if (!in_range_bytes.line_segment_intersection(ent_range_bytes)) {
            return OverlapStatus::NO_OVERLAP;
        }

        // -------- L2 prereqs: same axis layout? --------
        if (input.dtype != dtype || input.ndims != ndims || ndims == 0) {
            return OverlapStatus::OTHER;
        }
        for (uint32_t i = 0; i < ndims; i++) {
            if (input.strides[i] != strides[i]) return OverlapStatus::OTHER;
        }
        // strides[ndims-1] must be 1 and strides[i-1] must be an integer
        // multiple of strides[i] for the row-major reference-shape derivation
        // below to hold. This rejects slice-with-step (strides[d] != prev factor)
        // and any view chain that scrambles the axis order. (strides is
        // uint32_t with the > 0 invariant enforced at construction, so no
        // sign check needed.)
        if (strides[ndims - 1] != 1) return OverlapStatus::OTHER;
        for (uint32_t i = 1; i < ndims; i++) {
            if (strides[i - 1] % strides[i] != 0) return OverlapStatus::OTHER;
        }

        // Derive reference shape A from stride. By construction stride is
        // row-major over A: strides[i] = prod(A[i+1..ndims-1]). So
        //   A[i] = strides[i-1] / strides[i]   for i >= 1
        //   A[0] = (buffer.size / dtype_bytes) / strides[0]
        // input.buffer.size is the storage size; entry shares the same buffer
        // (debug-asserted by buffer.addr equality at the top), so we read it
        // from input rather than mirroring buffer.size into the entry.
        //
        // Note on buffer padding: runtime allocators may over-allocate
        // `buffer.size` (cache-line / 1024B alignment, ring-buffer slot
        // rounding, etc). When that happens, `numel_storage` is larger than
        // the true logical extent and `ref_shapes[0]` ends up generously over-
        // sized. This is intentional: ref_shapes is only used as an *upper
        // bound* in the in-bounds checks below; the actual overlap test (the
        // per-dim line-segment intersection on the real start_offset /
        // shapes / stride further down) is unaffected. A larger-than-truth
        // ref_shapes[0] simply makes the bounds check more permissive — it
        // can never cause a false NO_OVERLAP nor a false COVERED.
        uint32_t ref_shapes[RUNTIME_MAX_TENSOR_DIMS] = {};
        for (uint32_t i = 1; i < ndims; i++) {
            ref_shapes[i] = strides[i - 1] / strides[i];
        }
        const uint64_t elem_size = get_element_size(dtype);
        if (elem_size == 0) return OverlapStatus::OTHER;
        const uint64_t numel_storage = input.buffer.size / elem_size;
        const uint32_t stride0 = strides[0];  // > 0 by Tensor invariant
        if (numel_storage % stride0 != 0) return OverlapStatus::OTHER;
        ref_shapes[0] = static_cast<uint32_t>(numel_storage / stride0);

        // Decompose start_offset into row-major multi-dim offsets. By the same
        // relation strides[i] = prod(ref_shapes[i+1..]) so dividing by strides[i]
        // (no inner loop) yields each axis offset directly.
        uint32_t in_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
        uint32_t ent_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
        uint64_t in_remain = input.start_offset;
        uint64_t ent_remain = start_offset;
        for (uint32_t i = 0; i < ndims; i++) {
            const uint32_t s = strides[i];
            in_offsets[i] = static_cast<uint32_t>(in_remain / s);
            ent_offsets[i] = static_cast<uint32_t>(ent_remain / s);
            in_remain %= s;
            ent_remain %= s;
        }
        if (in_remain != 0 || ent_remain != 0) return OverlapStatus::OTHER;

        // Validate that each side fits within ref_shapes (defense in depth —
        // a well-formed view always satisfies this).
        for (uint32_t i = 0; i < ndims; i++) {
            if (static_cast<uint64_t>(in_offsets[i]) + input.shapes[i] > ref_shapes[i]) return OverlapStatus::OTHER;
            if (static_cast<uint64_t>(ent_offsets[i]) + shapes[i] > ref_shapes[i]) return OverlapStatus::OTHER;
        }

        // -------- L2 core: per-dim line-segment intersection --------
        bool input_contains_entry = true;
        for (uint32_t i = 0; i < ndims; i++) {
            Segment in_seg{in_offsets[i], static_cast<uint64_t>(in_offsets[i]) + input.shapes[i]};
            Segment ent_seg{ent_offsets[i], static_cast<uint64_t>(ent_offsets[i]) + shapes[i]};
            if (!in_seg.line_segment_intersection(ent_seg)) {
                return OverlapStatus::NO_OVERLAP;
            }
            if (!in_seg.contains(ent_seg)) {
                input_contains_entry = false;
            }
        }
        return input_contains_entry ? OverlapStatus::COVERED : OverlapStatus::OTHER;
    }
};

static_assert(sizeof(PTO2TensorMapEntry) == 128, "TensorMapEntry must be exactly 2 cache lines (128 bytes)");
static_assert(offsetof(PTO2TensorMapEntry, buffer_addr) == offsetof(Tensor, buffer.addr));
static_assert(offsetof(PTO2TensorMapEntry, producer_task_id) == offsetof(Tensor, owner_task_id));
static_assert(offsetof(PTO2TensorMapEntry, start_offset) == offsetof(Tensor, start_offset));
static_assert(offsetof(PTO2TensorMapEntry, version) == offsetof(Tensor, version));
static_assert(offsetof(PTO2TensorMapEntry, ndims) == offsetof(Tensor, ndims));
static_assert(offsetof(PTO2TensorMapEntry, dtype) == offsetof(Tensor, dtype));
static_assert(offsetof(PTO2TensorMapEntry, manual_dep) == offsetof(Tensor, manual_dep));
static_assert(offsetof(PTO2TensorMapEntry, is_contiguous) == offsetof(Tensor, is_contiguous));
static_assert(offsetof(PTO2TensorMapEntry, shapes) == offsetof(Tensor, shapes));
static_assert(
    offsetof(PTO2TensorMapEntry, prev_in_bucket) == 64, "TensorMapEntry must be exactly 2 cache lines (128 bytes)"
);

// =============================================================================
// TensorMap Lookup Chain Length Statistics (compile-time toggle)
// =============================================================================

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

    uint32_t get_task_local_id_slot(uint8_t ring_id, uint32_t task_local_id) const {
        return task_local_id & (task_window_sizes[ring_id] - 1);
    }

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
     * Phase 1: reserve every sub-region (buckets, entry_pool, free list, per-ring
     * task_entry_heads) on the supplied arena. Records the resulting offsets in
     * the returned layout descriptor. Must be called before the arena is
     * committed.
     */
    static PTO2TensorMapLayout reserve_layout(
        DeviceArena &arena, int32_t num_buckets, int32_t pool_size, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH]
    );

    /**
     * Same as reserve_layout() with default sizes (PTO2_TENSORMAP_NUM_BUCKETS,
     * PTO2_TENSORMAP_POOL_SIZE).
     */
    static PTO2TensorMapLayout
    reserve_layout_default(DeviceArena &arena, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH]);

    /**
     * Phase 3: bind region pointers and initialize state. The arena must already
     * be committed; layout must have been produced by reserve_layout() against
     * the same arena.
     */
    bool init_from_layout(const PTO2TensorMapLayout &layout, DeviceArena &arena);

    /**
     * Tear down state. Does not free memory — the arena owns the backing
     * buffer. Pointers are set to nullptr so accidental reuse traps.
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
     * Searches the hash table for matching regions and invokes the callback
     * for each overlapping valid entry.
     * Stale entries from different rings are skipped (not truncated).
     *
     * The callback receives (PTO2TensorMapEntry &, OverlapStatus) and should
     * return true to continue iteration, false to stop early. It is safe for
     * the callback to call remove_entry() on the current entry: next_in_bucket
     * is latched before invocation.
     *
     * @param tensor    Tensor to look up
     * @param on_match  Callback invoked for each overlapping entry
     */
    template <typename Fn>
    void lookup(const Tensor &tensor, Fn &&on_match) {
        uint32_t bucket_index = hash(tensor.buffer.addr);
        PTO2TensorMapEntry *cur_entry = buckets[bucket_index];

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
#if PTO2_TENSORMAP_PROFILING
                    g_lookup_overlap_hits++;
#endif
                    if (!on_match(*cur_entry, overlap_status)) {
#if PTO2_TENSORMAP_PROFILING
                        g_lookup_chain_total += chain_len;
                        if (chain_len > g_lookup_chain_max) g_lookup_chain_max = chain_len;
#endif
                        return;
                    }
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
    void sync_tensormap(PTO2TaskId task_id, int32_t sm_last_task_alive);
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
