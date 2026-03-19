/**
 * PTO Runtime2 - TensorMap Interface
 *
 * TensorMap provides producer lookup for dependency discovery:
 * - Maps Tensor -> producer task ID
 * - Used by pto2_submit_task() to find dependencies
 *
 * The facade owns:
 * - one owner shard per ring for same-ring history
 * - one fallback shard for external tensors and cross-ring modifiers
 *
 * Owner and fallback both execute the same template implementation.
 *
 * Key design features:
 * 1. Ring-buffer entry pool for O(1) append/free-list reuse
 * 2. Producer-driven lazy invalidation and cleanup
 * 3. OVERLAP DETECTION: detects dependencies for overlapping sub-regions
 *
 * CRITICAL: Hash only by base_ptr
 * ==============================
 * For overlap detection to work, ALL sub-regions of the same base tensor
 * MUST be in the SAME hash bucket. This allows lookup to compare all
 * potentially overlapping regions.
 *
 * Overlap detection: Two regions create a dependency if:
 *   1. Same base_ptr (raw tensor pointer)
 *   2. Their multi-dimensional ranges intersect
 */

#pragma once

#include "common.h"
#include "pto_runtime2_types.h"
#include "tensor.h"

struct PTO2OrchestratorState;

#ifndef PTO2_TENSORMAP_PROFILING
#define PTO2_TENSORMAP_PROFILING 0
#endif

#if PTO2_TENSORMAP_PROFILING
extern uint64_t g_lookup_chain_total;
extern uint64_t g_lookup_count;
extern int32_t  g_lookup_chain_max;
extern uint64_t g_lookup_overlap_checks;
extern uint64_t g_lookup_overlap_hits;
extern uint64_t g_insert_count;
#endif

enum class TensorMapStorageDomain : uint8_t {
    OWNER_MAP = 0,
    FALLBACK_MAP = 1,
};

/**
 * TensorMap entry structure.
 *
 * Cache line 1 keeps lookup-hot metadata plus overlap info.
 * Cache line 2 keeps intrusive links and slow-path offsets.
 *
 * When is_all_offset_zero is true, lookup touches only cache line 1.
 * Entry size: 128B (2 cache lines).
 */
struct alignas(64) PTO2TensorMapEntry {
    // === Cache line 1 (64B) — lookup hot path ===
    PTO2TensorMapEntry* next_in_bucket;        // 8B: next entry in hash bucket chain
    PTO2TaskId producer_task_id;               // 8B: raw (ring_id << 32) | local_id
    uint64_t buffer_addr;                      // 8B: tensor base address (hash key)
    int32_t version;                           // 4B: tensor version for overlap detection
    int32_t bucket_index;                      // 4B: bucket index (-1 if unlinked)
    uint16_t ndims;                            // 2B: number of dimensions
    uint8_t tensor_owner_ring;                 // 1B: tensor owner ring, or TENSOR_RING_ID_NONE
    TensorMapStorageDomain storage_domain;     // 1B: owner shard or fallback shard
    bool is_all_offset_zero;                   // 1B: fast-path flag
    bool with_alloc;                           // 1B: true=producer created a new runtime allocation; false=modifier or preallocated-output history
    uint32_t shapes[RUNTIME_MAX_TENSOR_DIMS];  // 20B: shape per dimension

    // === Cache line 2 (64B) — insert/remove/slow-path ===
    PTO2TensorMapEntry* prev_in_bucket;        // 8B: prev in hash bucket chain
    PTO2TensorMapEntry* next_in_task;          // 8B: next entry for same cleanup task
    PTO2TensorMapEntry* prev_in_task;          // 8B: prev entry for same cleanup task
    uint32_t offsets[RUNTIME_MAX_TENSOR_DIMS]; // 20B: only when !is_all_offset_zero

    /**
     * Copy overlap-relevant fields from a Tensor into this entry.
     */
    void copy_from_tensor(const Tensor& t) {
        buffer_addr = t.buffer.addr;
        version = t.version;
        ndims = static_cast<uint16_t>(t.ndims);
        is_all_offset_zero = t.is_all_offset_zero;
        for (uint32_t i = 0; i < t.ndims; i++) {
            shapes[i] = t.shapes[i];
        }
        if (!t.is_all_offset_zero) {
            for (uint32_t i = 0; i < t.ndims; i++) {
                offsets[i] = t.offsets[i];
            }
        }
    }

    uint8_t producer_ring() const {
        return pto2_task_id_ring(producer_task_id);
    }

    uint32_t producer_local() const {
        return pto2_task_id_local(producer_task_id);
    }

    uint8_t storage_ring() const {
        debug_assert(storage_domain == TensorMapStorageDomain::OWNER_MAP);
        return tensor_owner_ring;
    }

    /**
     * Check overlap between the input tensor and this entry.
     * Mirrors Tensor::is_overlap() logic but operates on entry fields directly.
     */
    OverlapStatus check_overlap(const Tensor& input) const {
        debug_assert(input.buffer.addr == buffer_addr);
        debug_assert(input.version >= version);
        if (input.version > version) {
            return OverlapStatus::OTHER;
        }

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

        bool contains = true;
        for (uint32_t i = 0; i < ndims; i++) {
            uint64_t in_off = input.is_all_offset_zero ? 0 : input.offsets[i];
            uint64_t ent_off = is_all_offset_zero ? 0 : offsets[i];
            Segment in_range{in_off, in_off + static_cast<uint64_t>(input.shapes[i])};
            Segment ent_range{ent_off, ent_off + static_cast<uint64_t>(shapes[i])};
            if (!in_range.line_segment_intersection(ent_range)) {
                return OverlapStatus::NO_OVERLAP;
            }
            if (!in_range.contains(ent_range)) {
                contains = false;
            }
        }
        return contains ? OverlapStatus::COVERED : OverlapStatus::OTHER;
    }
};

static_assert(sizeof(PTO2TensorMapEntry) == 128, "TensorMapEntry must be exactly 2 cache lines (128 bytes)");

#define PTO2_LOOKUP_MAX_RESULTS 16

/**
 * Stack-allocated lookup result buffer.
 *
 * Facade lookup appends results from owner shard first and fallback shard next.
 */
struct PTO2LookupResult {
    struct Entry {
        PTO2TensorMapEntry* entry;
        OverlapStatus overlap_status;
    };
    Entry entries[PTO2_LOOKUP_MAX_RESULTS];
    int32_t count{0};

    void push(PTO2TensorMapEntry* entry, OverlapStatus s) {
        if (count < PTO2_LOOKUP_MAX_RESULTS) {
            entries[count++] = {entry, s};
        }
    }
};

struct TensorMapInsertMeta {
    // Original tensor owner ring, or TENSOR_RING_ID_NONE for external tensors.
    uint8_t tensor_owner_ring{TENSOR_RING_ID_NONE};

    // Which shard actually stores this entry.
    TensorMapStorageDomain storage_domain{TensorMapStorageDomain::OWNER_MAP};

    // True when producer created a new runtime allocation for this history entry.
    bool with_alloc{false};
};

/**
 * Shared core for both owner and fallback storage.
 *
 * OwnerTensorMapShard:
 * - NumCleanupDomains = 1
 * - BreakOnStale = true
 *
 * FallbackTensorMapShard:
 * - NumCleanupDomains = PTO2_MAX_RING_DEPTH
 * - BreakOnStale = false
 */
template <int32_t NumCleanupDomains, bool BreakOnStale>
struct TensorMapShardImpl {
    static_assert(NumCleanupDomains > 0, "TensorMapShardImpl must have at least one cleanup domain");
    static constexpr int32_t kNumCleanupDomains = NumCleanupDomains;
    static constexpr bool kBreakOnStale = BreakOnStale;

    PTO2TensorMapEntry** buckets{nullptr};
    int32_t num_buckets{0};

    PTO2TensorMapEntry* entry_pool{nullptr};
    PTO2TensorMapEntry** free_entry_list{nullptr};
    int32_t pool_size{0};
    int32_t next_entry_idx{0};
    int32_t free_num{0};
    int32_t owner_ring_entry_counts[PTO2_MAX_RING_DEPTH]{};
    int32_t external_entry_count{0};

    // task_entry_heads[domain][local & (window_size - 1)] -> intrusive list head
    PTO2TensorMapEntry** task_entry_heads[NumCleanupDomains]{};
    int32_t task_window_sizes[NumCleanupDomains]{};
    int32_t last_task_alives[NumCleanupDomains]{};
    int32_t last_cleanup[NumCleanupDomains]{};

    /**
     * Initialize shard storage.
     *
     * num_buckets and each task_window_sizes[d] must be powers of two.
     */
    bool init(int32_t num_buckets, int32_t pool_size, const int32_t task_window_sizes[]);
    void destroy();

    /**
     * Mirror the latest last_task_alive for one cleanup domain.
     */
    void sync_validity(int32_t cleanup_domain, int32_t last_task_alive);

    /**
     * Decide whether cleanup should run for [last_cleanup, new_last_task_alive).
     */
    bool need_cleanup(int32_t cleanup_domain, int32_t new_last_task_alive) const;

    /**
     * Lookup producer history for one shard.
     *
     * Results are appended into the caller-provided buffer.
     */
    void lookup(const Tensor& tensor, PTO2LookupResult& result);

    /**
     * For fallback shards, lazily refresh only the producer domains that appear
     * in the target bucket before running lookup.
     */
    void refresh_lookup_domains(const Tensor& tensor, PTO2OrchestratorState* orch);

    /**
     * Insert a new history entry into the shard selected by the facade.
     */
    PTO2TensorMapEntry* insert(const Tensor& tensor, PTO2TaskId producer_task_id, const TensorMapInsertMeta& meta);

    /**
     * Remove one entry from both bucket chain and task chain in O(1).
     */
    void remove_entry(PTO2TensorMapEntry& entry);

    /**
     * Cleanup retired producer tasks in one cleanup domain.
     */
    void cleanup_range(int32_t cleanup_domain, int32_t old_last_task_alive, int32_t new_last_task_alive);
    int32_t valid_count() const;
    void print_stats(const char* label) const;
    bool maybe_has_owner_ring(uint8_t owner_ring) const;

private:
    /**
     * Compute hash for tensor base address.
     */
    uint32_t hash(uint64_t key) const;

    int32_t cleanup_domain_of(PTO2TaskId producer_task_id) const;
    int32_t cleanup_domain_of(const PTO2TensorMapEntry& entry) const;
    int32_t lifecycle_local_of(PTO2TaskId producer_task_id) const;
    int32_t lifecycle_local_of(const PTO2TensorMapEntry& entry) const;

    /**
     * Shared validity rule:
     * producer_local >= last_task_alives[cleanup_domain_of(entry)]
     */
    bool entry_valid(const PTO2TensorMapEntry& entry) const;
    PTO2TensorMapEntry* new_entry();
    void account_entry_added(const PTO2TensorMapEntry& entry);
    void account_entry_removed(const PTO2TensorMapEntry& entry);
    void unlink_from_bucket(PTO2TensorMapEntry& entry);
    void unlink_from_task(PTO2TensorMapEntry& entry);
    void reclaim_entry(PTO2TensorMapEntry& entry);
};

using OwnerTensorMapShard = TensorMapShardImpl<1, true>;
using FallbackTensorMapShard = TensorMapShardImpl<PTO2_MAX_RING_DEPTH, false>;

/**
 * Facade that routes requests across owner shards and the fallback shard.
 */
struct PTO2TensorMap {
    OwnerTensorMapShard owner_shards[PTO2_MAX_RING_DEPTH];
    FallbackTensorMapShard fallback_shard;

    PTO2OrchestratorState* orch{nullptr};

    bool init(int32_t num_buckets, int32_t pool_size, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH]);
    bool init_default(const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH]);
    void destroy();

    /**
     * internal tensor: owner shard first, then fallback shard
     * external tensor: fallback shard only
     */
    void lookup(const Tensor& tensor, PTO2LookupResult& result);

    /**
     * OUTPUT or INOUT routing:
     * - same-ring internal history -> owner shard
     * - cross-ring internal INOUT -> fallback shard
     * - external history -> fallback shard
     *
     * Internal OUTPUT is fail-fast if tensor owner ring != producer ring.
     */
    void insert(const Tensor& tensor, PTO2TaskId producer_task_id, PTOParamType param_type, bool with_alloc);

    /**
     * Remove by real storage location, not by producer ring or tensor owner alone.
     */
    void remove_entry(PTO2TensorMapEntry& entry);

    void print_stats() const;
    int32_t valid_count() const;

    /**
     * Sync TensorMap validity threshold from shared memory.
     *
     * Signature is kept stable for orchestrator call sites. The implementation
     * refreshes the submit ring eagerly. Cross-ring fallback freshness is
     * recovered lazily during lookup by reading only the producer domains that
     * actually appear in the touched bucket.
     */
    void sync_tensormap(uint8_t ring_id, int32_t sm_last_task_alive);
};

#if PTO2_TENSORMAP_PROFILING
struct PTO2TensorMapProfilingData {
    uint64_t lookup_chain_total;
    uint64_t lookup_count;
    int32_t  lookup_chain_max;
    uint64_t overlap_checks;
    uint64_t overlap_hits;
    uint64_t insert_count;
};

PTO2TensorMapProfilingData pto2_tensormap_get_profiling();
#endif
