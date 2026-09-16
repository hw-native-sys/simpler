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
 * host_build_graph TensorMap interface
 *
 * TensorMap provides producer lookup for dependency discovery:
 * - Maps simpler::hbg::Tensor -> producer task ID
 * - Used by rt_submit_task() to find dependencies
 *
 * host_build_graph runs its orchestrator on the host, so this map is host-only
 * state: it owns its four arrays outright rather than addressing them as offsets
 * into a device-shaped arena. Nothing here is copied to the device.
 *
 * Key design features:
 * 1. Fixed-capacity entry pool (no per-entry malloc/free)
 * 2. Task completion does not retire entries; a producer stays visible until
 *    dependency computation explicitly removes it as semantically covered
 * 3. Per-task entry tracking for explicit removal
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

#include <memory>

#include "assert_compat.h"
#include "host_build_graph/task_id.h"
#include "host_build_graph/tensor_create_info.h"
#include "profiling_config.h"
#include "tensor.h"

// Overlap geometry types. Relocated here from tensor.h: they are used only by
// the runtime's overlap-detection / dependency machinery, not by the
// wire/host-facing simpler::hbg::Tensor definition.
enum class OverlapStatus {
    NO_OVERLAP,
    COVERED,
    OTHER,
};

struct Segment {
    uint64_t begin;
    uint64_t end;

    bool line_segment_intersection(const Segment &other) const { return end > other.begin && other.end > begin; }
    bool contains(const Segment &other) const { return begin <= other.begin && other.end <= end; }
};

// Slot index for `addr` in a power-of-two table of 2^slot_bits entries. Shared by every
// table in this runtime that is keyed by a buffer address: this map's buckets, and the
// boundary alias partition in the host orchestrator.
//
// Multiplicative hash on the golden-ratio constant. The multiply mixes all of the input's
// bits into the *high* bits of the product, and taking the top `slot_bits` is what makes
// aligned keys distribute -- every buffer here is at least PACKED_OUTPUT_ALIGN-aligned, so
// its low bits are constant and masking them would collide every key onto one slot.
//
// Returns the slot and not the hash, so that truncation cannot be done wrong by a caller:
// there is one correct way to narrow this product and it lives here.
inline uint32_t addr_to_slot(uint64_t addr, uint32_t slot_bits) {
    return static_cast<uint32_t>((addr * 0x9E3779B97F4A7C15ULL) >> (64 - slot_bits));
}

// TensorMap Lookup Profiling (must precede inline lookup/insert methods).
#if SIMPLER_TENSORMAP_PROFILING
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
 * Cache line 1 (64B, lookup hot path) mirrors simpler::hbg::Tensor cache line 1 byte-for-byte
 * from byte 16 onward, so that `memcpy(this, &tensor, 64)` populates everything
 * we need for overlap checks. Bytes [0, 16) carry entry-only fields (hash
 * bucket head + chain pointer) that overlap simpler::hbg::Tensor::buffer (addr in [0, 8) is
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
 * Entry size: 128B (2 cache lines), matches simpler::hbg::Tensor.
 */
struct alignas(64) ChipTensorMapEntry {
    // === Cache line 1 (64B) — lookup hot path; mirrors simpler::hbg::Tensor line 1 from byte 16 ===
    uint64_t buffer_addr;  // 8B [0, 8):   tensor base address (hash key, mirrors simpler::hbg::Tensor::buffer.addr)
    ChipTensorMapEntry
        *next_in_bucket;  // 8B [8, 16):  next entry in hash bucket chain (overlays simpler::hbg::Tensor::buffer.size)
    TaskId producer_task_id;           // 8B [16,24):  mirrors simpler::hbg::Tensor::owner_task_id slot
    uint64_t start_offset;             // 8B [24,32):  mirrors simpler::hbg::Tensor::start_offset (element offset)
    int32_t version;                   // 4B [32,36):  mirrors simpler::hbg::Tensor::version
    uint8_t ndims;                     // 1B [36,37):  mirrors simpler::hbg::Tensor::ndims
    DataType dtype;                    // 1B [37,38):  mirrors simpler::hbg::Tensor::dtype
    bool manual_dep;                   // 1B [38,39):  mirrors simpler::hbg::Tensor::manual_dep
    bool is_contiguous;                // 1B [39,40):  mirrors simpler::hbg::Tensor::is_contiguous
    uint8_t __padding1__[4];           // 4B [40,44):  spans simpler::hbg::Tensor::address_space, which an
                                       //              entry never reads, and the padding after it
    uint32_t shapes[MAX_TENSOR_DIMS];  // 20B [44,64): mirrors simpler::hbg::Tensor::shapes

    // === Cache line 2 (64B) — chain manipulation + non-contiguous overlap data ===
    ChipTensorMapEntry *prev_in_bucket;  // 8B [64, 72)
    ChipTensorMapEntry *next_in_task;    // 8B [72, 80)
    ChipTensorMapEntry *prev_in_task;    // 8B [80, 88)
    int32_t bucket_index;                // 4B [88, 92): -1 when unlinked
    uint32_t __padding2__;               // 4B [92, 96)
    uint64_t extent_elem_cache;          // 8B [96,104): non-contiguous extent (mirrors simpler::hbg::Tensor)
    uint32_t strides[MAX_TENSOR_DIMS];   // 20B [104,124): element strides, mirrors simpler::hbg::Tensor::strides
    uint8_t __padding3__[4];             // 4B [124,128)

    /**
     * Copy overlap-relevant fields from a simpler::hbg::Tensor into this entry.
     *
     * 64B memcpy of simpler::hbg::Tensor cache line 1 populates buffer_addr (byte [0,8)),
     * producer_task_id, start_offset, version, ndims, dtype, manual_dep,
     * is_contiguous and shapes[]. Byte [8,16) holds simpler::hbg::Tensor::buffer.size in
     * the source and gets written into next_in_bucket; that's harmless
     * because link_entry() overwrites next_in_bucket immediately after.
     *
     * Cache line 2 (stride / extent_elem_cache) is derived from line 1 when
     * the source is canonically contiguous (is_contiguous && start_offset==0),
     * so the producer simpler::hbg::Tensor's cache line 2 stays cold during insert. Only
     * non-contiguous producers pay one extra line 2 read.
     */
    void copy_from_tensor(const simpler::hbg::Tensor &tensor) {
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
    OverlapStatus check_overlap(const simpler::hbg::Tensor &input) const {
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
        uint32_t ref_shapes[MAX_TENSOR_DIMS] = {};
        for (uint32_t i = 1; i < ndims; i++) {
            ref_shapes[i] = strides[i - 1] / strides[i];
        }
        const uint64_t elem_size = get_element_size(dtype);
        if (elem_size == 0) return OverlapStatus::OTHER;
        const uint64_t numel_storage = input.buffer.size / elem_size;
        const uint32_t stride0 = strides[0];  // > 0 by simpler::hbg::Tensor invariant
        if (numel_storage % stride0 != 0) return OverlapStatus::OTHER;
        ref_shapes[0] = static_cast<uint32_t>(numel_storage / stride0);

        // Decompose start_offset into row-major multi-dim offsets. By the same
        // relation strides[i] = prod(ref_shapes[i+1..]) so dividing by strides[i]
        // (no inner loop) yields each axis offset directly.
        uint32_t in_offsets[MAX_TENSOR_DIMS] = {};
        uint32_t ent_offsets[MAX_TENSOR_DIMS] = {};
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

static_assert(sizeof(ChipTensorMapEntry) == 128, "TensorMapEntry must be exactly 2 cache lines (128 bytes)");
static_assert(offsetof(ChipTensorMapEntry, buffer_addr) == offsetof(simpler::hbg::Tensor, buffer.addr));
static_assert(offsetof(ChipTensorMapEntry, producer_task_id) == offsetof(simpler::hbg::Tensor, owner_task_id));
static_assert(offsetof(ChipTensorMapEntry, start_offset) == offsetof(simpler::hbg::Tensor, start_offset));
static_assert(offsetof(ChipTensorMapEntry, version) == offsetof(simpler::hbg::Tensor, version));
static_assert(offsetof(ChipTensorMapEntry, ndims) == offsetof(simpler::hbg::Tensor, ndims));
static_assert(offsetof(ChipTensorMapEntry, dtype) == offsetof(simpler::hbg::Tensor, dtype));
static_assert(offsetof(ChipTensorMapEntry, manual_dep) == offsetof(simpler::hbg::Tensor, manual_dep));
static_assert(offsetof(ChipTensorMapEntry, is_contiguous) == offsetof(simpler::hbg::Tensor, is_contiguous));
static_assert(offsetof(ChipTensorMapEntry, shapes) == offsetof(simpler::hbg::Tensor, shapes));
static_assert(
    offsetof(ChipTensorMapEntry, prev_in_bucket) == 64, "TensorMapEntry must be exactly 2 cache lines (128 bytes)"
);

// =============================================================================
// TensorMap Lookup Chain Length Statistics (compile-time toggle)
// =============================================================================

/**
 * TensorMap structure
 *
 * Hash table with a fixed-capacity entry pool and no watermark invalidation.
 * Owns its four arrays; init() sizes them and leaves the map empty.
 */
struct ChipTensorMap {
    // Hash table buckets (fixed size, power of 2). An empty bucket is nullptr.
    std::unique_ptr<ChipTensorMapEntry *[]> buckets;
    int32_t num_buckets{0};  // Must be power of 2 for fast modulo

    // Entry pool: bump allocation plus reuse of explicitly removed entries. A
    // linked entry is reached by pointer, so the pool is allocated once at
    // pool_size and never resized.
    std::unique_ptr<ChipTensorMapEntry[]> entry_pool;
    std::unique_ptr<ChipTensorMapEntry *[]> free_entry_list;
    int32_t pool_size{0};       // Total pool capacity
    int32_t next_entry_idx{0};  // id when next entry insert
    int32_t free_num{0};        // free entry number in entry pool

    // Per-task entry tracking for O(1) unlinking of covered producers. A task id
    // is its own slot index, so this is indexed by local_id directly.
    std::unique_ptr<ChipTensorMapEntry *[]> task_entry_heads;
    int32_t max_tasks{0};  // Slots task_entry_heads is dimensioned for

    // Pool occupancy, read by the tensormap-exhaustion diagnostic and by the
    // Graph recording's capacity precheck.
    int32_t current_used() const { return next_entry_idx - free_num; }
    int32_t pool_capacity() const { return pool_size; }
    int32_t free_entries() const { return pool_size - current_used(); }

    // new_entry allocates a slot and initializes only its linkage (bucket_index
    // and the four link pointers) to the clean unlinked state; insert() assigns
    // the tensor attributes and producer_task_id.
    ChipTensorMapEntry *new_entry() {
        if (free_num > 0) {
            ChipTensorMapEntry *res = free_entry_list[--free_num];
            debug_assert(res->bucket_index == -1);
            return res;
        }
        always_assert(next_entry_idx < pool_size);
        ChipTensorMapEntry *res = &entry_pool[next_entry_idx++];
        // Init-on-write: the pool is not pre-zeroed (init() skips
        // the O(pool_size) memset), so put this fresh slot into the same clean
        // unlinked state free_entry() leaves recycled slots in. The insert path
        // overwrites the remaining fields exactly as it does for a recycled slot,
        // so a fresh slot is indistinguishable from a reused one.
        res->bucket_index = -1;
        res->next_in_bucket = nullptr;
        res->prev_in_bucket = nullptr;
        res->next_in_task = nullptr;
        res->prev_in_task = nullptr;
        return res;
    }

    void free_entry(ChipTensorMapEntry &entry) {
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
     * Allocate the four arrays and leave the map empty. num_buckets must be a
     * power of two; `new_max_tasks` is the number of task chains to reserve, one
     * per task id the caller can place.
     *
     * Returns false when an allocation fails, so a caller that still has an
     * alternative path can take it rather than proceed without a hazard map.
     *
     * Clearing is O(num_buckets + max_tasks), not O(pool_size): the entry
     * pool is left uninitialized and new_entry() puts each slot into the clean
     * unlinked state on first use, and free_entry_list is a stack meaningful only
     * below free_num.
     */
    bool init(int32_t new_num_buckets, int32_t new_pool_size, int32_t new_max_tasks);

    /**
     * Empty an already-initialized map without touching its allocations.
     *
     * Same post-state as init(): every bucket and task-chain head null, both pool
     * cursors at zero, the entry pool left to init-on-write. Costs
     * O(num_buckets + max_tasks) stores against pages that are already
     * resident, where init() pays the allocation and the first touch of each. A
     * caller that records one body after another on the same map uses this.
     *
     * Keeps the bucket count and task-chain count init() reserved. Taking a new
     * count here would have to be checked against the reserved length rather than
     * the current one, and nothing tracks the reserved length once a smaller count
     * has been set -- so the sizes stay init()'s and this takes no argument.
     */
    void reset();

    /**
     * Same as init() with default sizes (CHIP_TENSORMAP_NUM_BUCKETS,
     * CHIP_TENSORMAP_POOL_SIZE).
     */
    bool init_default(int32_t new_max_tasks);

    /**
     * Lookup producer for a tensor region
     *
     * Searches the hash table for matching regions and invokes the callback
     * for each overlapping entry.
     *
     * The callback receives (ChipTensorMapEntry &, OverlapStatus) and should
     * return true to continue iteration, false to stop early. It is safe for
     * the callback to call remove_entry() on the current entry: next_in_bucket
     * is latched before invocation.
     *
     * @param tensor    simpler::hbg::Tensor to look up
     * @param on_match  Callback invoked for each overlapping entry
     */
    template <typename Fn>
    void lookup(const simpler::hbg::Tensor &tensor, Fn &&on_match) {
        uint32_t bucket_index = hash(tensor.buffer.addr);
        ChipTensorMapEntry *cur_entry = buckets[bucket_index];

#if SIMPLER_TENSORMAP_PROFILING
        g_lookup_count++;
        int32_t chain_len = 0;
#endif

        while (cur_entry != nullptr) {
            ChipTensorMapEntry *next_entry = cur_entry->next_in_bucket;

#if SIMPLER_TENSORMAP_PROFILING
            chain_len++;
#endif
            // Check if regions OVERLAP (not just exact match)
            // Since we hash only by base_ptr, all entries in this bucket have
            // potential to overlap. We must check actual byte-range overlap.
            if (tensor.buffer.addr == cur_entry->buffer_addr) {
#if SIMPLER_TENSORMAP_PROFILING
                g_lookup_overlap_checks++;
#endif
                auto overlap_status = cur_entry->check_overlap(tensor);
                if (overlap_status != OverlapStatus::NO_OVERLAP) {
#if SIMPLER_TENSORMAP_PROFILING
                    g_lookup_overlap_hits++;
#endif
                    if (!on_match(*cur_entry, overlap_status)) {
#if SIMPLER_TENSORMAP_PROFILING
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
#if SIMPLER_TENSORMAP_PROFILING
        g_lookup_chain_total += chain_len;
        if (chain_len > g_lookup_chain_max) g_lookup_chain_max = chain_len;
#endif
    }

    /**
     * Insert a new entry (called when task produces output)
     *
     * Allocates from the fixed pool or its explicit-removal free list. A live
     * entry is never overwritten.
     * Inserts at head of hash bucket chain (maintains task_id ordering).
     *
     * @param tensor            simpler::hbg::Tensor produced
     * @param producer_task_id  Task ID of producer
     */
    void insert(const simpler::hbg::Tensor &tensor, TaskId producer_task_id) {
        ChipTensorMapEntry *entry = new_entry();
        entry->copy_from_tensor(tensor);
        link_entry(entry, tensor.buffer.addr, producer_task_id);
    }

    // =============================================================================
    // Internal Helpers (exposed for testing)
    // =============================================================================

    /**
     * Compute hash for tensor addr
     *
     * addr_to_slot over this table's bucket count, which is a power of two so its
     * trailing-zero count is log2 of it.
     */
    uint32_t hash(uint64_t key) { return addr_to_slot(key, static_cast<uint32_t>(__builtin_ctz(num_buckets))); }

    /**
     * Link an initialized entry into bucket and task chains.
     */
    void link_entry(ChipTensorMapEntry *entry, uint64_t addr, TaskId producer_task_id) {
#if SIMPLER_TENSORMAP_PROFILING
        g_insert_count++;
#endif
        uint32_t bucket_index = hash(addr);
        const int32_t task_slot = producer_task_id.local_id();
        // A producer's low id field is a task chain index directly, so the id space a
        // caller inserts under has to be the one this map was dimensioned for: a
        // whole-run map takes task capacity, a Graph recording's takes MAX_IN_GRAPH_TASKS.
        debug_assert(task_slot >= 0 && task_slot < max_tasks);

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
        entry->next_in_task = task_entry_heads[task_slot];
        entry->prev_in_task = nullptr;
        if (entry->next_in_task != nullptr) {
            entry->next_in_task->prev_in_task = entry;
        }
        task_entry_heads[task_slot] = entry;
    }

    void remove_entry(ChipTensorMapEntry &entry) {
        remove_from_task(entry);
        free_entry(entry);
    }

    /**
     * Remove an entry from its task chain (O(1) with prev pointer). Dependency
     * computation calls this before freeing a producer made redundant by a
     * covering input.
     */
    void remove_from_task(ChipTensorMapEntry &entry) {
        always_assert(entry.bucket_index != -1);  // must still be in a bucket
        // Update predecessor's next pointer (O(1) via prev_in_task)
        if (entry.prev_in_task == nullptr) {
            // Entry is the head of its task chain, update task_entry_heads
            int32_t local_id = entry.producer_task_id.local_id();
            const int32_t task_slot = local_id;
            debug_assert(task_slot >= 0 && task_slot < max_tasks);
            task_entry_heads[task_slot] = entry.next_in_task;
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
};

#if SIMPLER_TENSORMAP_PROFILING
struct ChipTensorMapProfilingData {
    uint64_t lookup_chain_total;
    uint64_t lookup_count;
    int32_t lookup_chain_max;
    uint64_t overlap_checks;
    uint64_t overlap_hits;
    uint64_t insert_count;
};

ChipTensorMapProfilingData chip_tensormap_get_profiling();
#endif
