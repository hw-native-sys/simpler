/**
 * PTO Runtime2 - TensorMap Implementation
 */

#include "pto_tensormap.h"

#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "common/unified_log.h"
#include "pto_orchestrator.h"

#if PTO2_TENSORMAP_PROFILING
uint64_t g_lookup_chain_total = 0;
uint64_t g_lookup_count = 0;
int32_t  g_lookup_chain_max = 0;
uint64_t g_lookup_overlap_checks = 0;
uint64_t g_lookup_overlap_hits = 0;
uint64_t g_insert_count = 0;
#endif

namespace {

bool is_power_of_two(int32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

bool tensor_has_valid_owner_ring(const Tensor& tensor) {
    return tensor.ring_id == TENSOR_RING_ID_NONE || tensor.ring_id < PTO2_MAX_RING_DEPTH;
}

}  // namespace

template <int32_t NumCleanupDomains, bool BreakOnStale>
bool TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::init(
    int32_t new_num_buckets, int32_t new_pool_size, const int32_t new_task_window_sizes[]) {
    if (!is_power_of_two(new_num_buckets)) {
        return false;
    }
    for (int d = 0; d < NumCleanupDomains; d++) {
        if (!is_power_of_two(new_task_window_sizes[d])) {
            return false;
        }
    }

    num_buckets = new_num_buckets;
    pool_size = new_pool_size;

    buckets = static_cast<PTO2TensorMapEntry**>(malloc(num_buckets * sizeof(PTO2TensorMapEntry*)));
    if (!buckets) {
        return false;
    }
    for (int32_t i = 0; i < num_buckets; i++) {
        buckets[i] = nullptr;
    }

    entry_pool = static_cast<PTO2TensorMapEntry*>(
        aligned_alloc(alignof(PTO2TensorMapEntry), pool_size * sizeof(PTO2TensorMapEntry)));
    if (!entry_pool) {
        destroy();
        return false;
    }
    memset(entry_pool, 0, pool_size * sizeof(PTO2TensorMapEntry));

    free_entry_list = static_cast<PTO2TensorMapEntry**>(calloc(pool_size, sizeof(PTO2TensorMapEntry*)));
    if (!free_entry_list) {
        destroy();
        return false;
    }

    next_entry_idx = 0;
    free_num = 0;
    external_entry_count = 0;
    for (int32_t r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        owner_ring_entry_counts[r] = 0;
    }

    for (int32_t i = 0; i < pool_size; i++) {
        entry_pool[i].bucket_index = -1;
        entry_pool[i].next_in_bucket = nullptr;
        entry_pool[i].prev_in_bucket = nullptr;
        entry_pool[i].next_in_task = nullptr;
        entry_pool[i].prev_in_task = nullptr;
        entry_pool[i].producer_task_id = PTO2TaskId{};
        entry_pool[i].tensor_owner_ring = TENSOR_RING_ID_NONE;
        entry_pool[i].storage_domain = TensorMapStorageDomain::OWNER_MAP;
        entry_pool[i].with_alloc = false;
    }

    for (int d = 0; d < NumCleanupDomains; d++) {
        task_entry_heads[d] = static_cast<PTO2TensorMapEntry**>(
            malloc(new_task_window_sizes[d] * sizeof(PTO2TensorMapEntry*)));
        if (!task_entry_heads[d]) {
            destroy();
            return false;
        }
        for (int32_t i = 0; i < new_task_window_sizes[d]; i++) {
            task_entry_heads[d][i] = nullptr;
        }
        task_window_sizes[d] = new_task_window_sizes[d];
        last_task_alives[d] = 0;
        last_cleanup[d] = 0;
    }

    return true;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
void TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::destroy() {
    if (buckets) {
        free(buckets);
        buckets = nullptr;
    }
    if (entry_pool) {
        free(entry_pool);
        entry_pool = nullptr;
    }
    if (free_entry_list) {
        free(free_entry_list);
        free_entry_list = nullptr;
    }
    for (int d = 0; d < NumCleanupDomains; d++) {
        if (task_entry_heads[d]) {
            free(task_entry_heads[d]);
            task_entry_heads[d] = nullptr;
        }
        task_window_sizes[d] = 0;
        last_task_alives[d] = 0;
        last_cleanup[d] = 0;
    }
    num_buckets = 0;
    pool_size = 0;
    next_entry_idx = 0;
    free_num = 0;
    external_entry_count = 0;
    for (int32_t r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        owner_ring_entry_counts[r] = 0;
    }
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
void TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::sync_validity(
    int32_t cleanup_domain, int32_t last_task_alive) {
    always_assert(cleanup_domain >= 0 && cleanup_domain < NumCleanupDomains);
    last_task_alives[cleanup_domain] = last_task_alive;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
bool TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::need_cleanup(
    int32_t cleanup_domain, int32_t new_last_task_alive) const {
    always_assert(cleanup_domain >= 0 && cleanup_domain < NumCleanupDomains);
    int32_t old_last_cleanup = last_cleanup[cleanup_domain];
    if (new_last_task_alive <= old_last_cleanup) {
        return false;
    }
    if (new_last_task_alive - old_last_cleanup >= PTO2_TENSORMAP_CLEANUP_INTERVAL) {
        return true;
    }
    return free_num < PTO2_TENSORMAP_CLEANUP_INTERVAL;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
uint32_t TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::hash(uint64_t key) const {
    key = key ^ (key >> 16);
    key = key ^ (key >> 32);
    return static_cast<uint32_t>(key & (num_buckets - 1));
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
int32_t TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::cleanup_domain_of(
    PTO2TaskId producer_task_id) const {
    if constexpr (NumCleanupDomains == 1) {
        (void)producer_task_id;
        return 0;
    } else {
        return static_cast<int32_t>(pto2_task_id_ring(producer_task_id));
    }
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
int32_t TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::cleanup_domain_of(
    const PTO2TensorMapEntry& entry) const {
    if constexpr (NumCleanupDomains == 1) {
        (void)entry;
        return 0;
    } else {
        return static_cast<int32_t>(entry.producer_ring());
    }
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
int32_t TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::lifecycle_local_of(
    PTO2TaskId producer_task_id) const {
    return static_cast<int32_t>(pto2_task_id_local(producer_task_id));
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
int32_t TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::lifecycle_local_of(
    const PTO2TensorMapEntry& entry) const {
    return static_cast<int32_t>(entry.producer_local());
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
bool TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::entry_valid(
    const PTO2TensorMapEntry& entry) const {
    int32_t cleanup_domain = cleanup_domain_of(entry);
    always_assert(cleanup_domain >= 0 && cleanup_domain < NumCleanupDomains);
    return lifecycle_local_of(entry) >= last_task_alives[cleanup_domain];
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
PTO2TensorMapEntry* TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::new_entry() {
    if (free_num > 0) {
        PTO2TensorMapEntry* res = free_entry_list[--free_num];
        debug_assert(res->bucket_index == -1);
        return res;
    }
    always_assert(next_entry_idx < pool_size);
    PTO2TensorMapEntry* res = &entry_pool[next_entry_idx++];
    debug_assert(res->bucket_index == -1);
    return res;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
void TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::account_entry_added(
    const PTO2TensorMapEntry& entry) {
    if (entry.tensor_owner_ring == TENSOR_RING_ID_NONE) {
        external_entry_count++;
        return;
    }
    always_assert(entry.tensor_owner_ring < PTO2_MAX_RING_DEPTH);
    owner_ring_entry_counts[entry.tensor_owner_ring]++;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
void TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::account_entry_removed(
    const PTO2TensorMapEntry& entry) {
    if (entry.tensor_owner_ring == TENSOR_RING_ID_NONE) {
        always_assert(external_entry_count > 0);
        external_entry_count--;
        return;
    }
    always_assert(entry.tensor_owner_ring < PTO2_MAX_RING_DEPTH);
    always_assert(owner_ring_entry_counts[entry.tensor_owner_ring] > 0);
    owner_ring_entry_counts[entry.tensor_owner_ring]--;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
void TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::unlink_from_bucket(
    PTO2TensorMapEntry& entry) {
    always_assert(entry.bucket_index != -1);
    if (entry.prev_in_bucket == nullptr) {
        buckets[entry.bucket_index] = entry.next_in_bucket;
    } else {
        entry.prev_in_bucket->next_in_bucket = entry.next_in_bucket;
    }
    if (entry.next_in_bucket != nullptr) {
        entry.next_in_bucket->prev_in_bucket = entry.prev_in_bucket;
    }
    entry.next_in_bucket = nullptr;
    entry.prev_in_bucket = nullptr;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
void TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::unlink_from_task(
    PTO2TensorMapEntry& entry) {
    int32_t cleanup_domain = cleanup_domain_of(entry);
    always_assert(cleanup_domain >= 0 && cleanup_domain < NumCleanupDomains);
    int32_t local_id = lifecycle_local_of(entry);
    int32_t task_slot = local_id & (task_window_sizes[cleanup_domain] - 1);
    if (entry.prev_in_task == nullptr) {
        task_entry_heads[cleanup_domain][task_slot] = entry.next_in_task;
    } else {
        entry.prev_in_task->next_in_task = entry.next_in_task;
    }
    if (entry.next_in_task != nullptr) {
        entry.next_in_task->prev_in_task = entry.prev_in_task;
    }
    entry.next_in_task = nullptr;
    entry.prev_in_task = nullptr;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
void TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::reclaim_entry(
    PTO2TensorMapEntry& entry) {
    always_assert(entry.bucket_index != -1);
    account_entry_removed(entry);
    unlink_from_bucket(entry);
    free_entry_list[free_num++] = &entry;
    entry.bucket_index = -1;
    entry.next_in_bucket = nullptr;
    entry.prev_in_bucket = nullptr;
    entry.next_in_task = nullptr;
    entry.prev_in_task = nullptr;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
void TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::lookup(
    const Tensor& tensor, PTO2LookupResult& result) {
    uint32_t bucket_index = hash(tensor.buffer.addr);
    PTO2TensorMapEntry* cur_entry = buckets[bucket_index];

#if PTO2_TENSORMAP_PROFILING
    g_lookup_count++;
    int32_t chain_len = 0;
#endif

    while (cur_entry != nullptr) {
        PTO2TensorMapEntry* next_entry = cur_entry->next_in_bucket;
        if (next_entry) __builtin_prefetch(next_entry, 0, 0);

#if PTO2_TENSORMAP_PROFILING
        chain_len++;
#endif

        if (!entry_valid(*cur_entry)) {
            if constexpr (BreakOnStale) {
                break;
            }
            // Fallback chains cannot break on stale, so prune dead history
            // here to avoid rescanning it on later lookups.
            remove_entry(*cur_entry);
            cur_entry = next_entry;
            continue;
        }

        if (tensor.buffer.addr == cur_entry->buffer_addr) {
            if (next_entry) {
                PTO2TensorMapEntry* next_next = next_entry->next_in_bucket;
                if (next_next) __builtin_prefetch(next_next, 0, 0);
            }
#if PTO2_TENSORMAP_PROFILING
            g_lookup_overlap_checks++;
#endif
            OverlapStatus overlap_status = cur_entry->check_overlap(tensor);
            if (overlap_status != OverlapStatus::NO_OVERLAP) {
                result.push(cur_entry, overlap_status);
#if PTO2_TENSORMAP_PROFILING
                g_lookup_overlap_hits++;
#endif
            }
        }

        cur_entry = next_entry;
    }

#if PTO2_TENSORMAP_PROFILING
    g_lookup_chain_total += chain_len;
    if (chain_len > g_lookup_chain_max) g_lookup_chain_max = chain_len;
#endif
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
void TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::refresh_lookup_domains(
    const Tensor& tensor, PTO2OrchestratorState* orch) {
    if constexpr (BreakOnStale) {
        (void)tensor;
        (void)orch;
        return;
    }

    if (orch == nullptr || orch->sm_handle == nullptr) {
        return;
    }

    bool seen[NumCleanupDomains] = {};
    int32_t new_alives[NumCleanupDomains] = {};
    uint32_t bucket_index = hash(tensor.buffer.addr);
    PTO2TensorMapEntry* cur_entry = buckets[bucket_index];
    while (cur_entry != nullptr) {
        if (cur_entry->buffer_addr == tensor.buffer.addr) {
            int32_t cleanup_domain = cleanup_domain_of(*cur_entry);
            if (!seen[cleanup_domain]) {
                seen[cleanup_domain] = true;
                new_alives[cleanup_domain] =
                    orch->sm_handle->header->rings[cleanup_domain].fc.last_task_alive.load(std::memory_order_acquire);
            }
        }
        cur_entry = cur_entry->next_in_bucket;
    }

    for (int32_t cleanup_domain = 0; cleanup_domain < NumCleanupDomains; cleanup_domain++) {
        if (!seen[cleanup_domain]) {
            continue;
        }
        sync_validity(cleanup_domain, new_alives[cleanup_domain]);
        if (need_cleanup(cleanup_domain, new_alives[cleanup_domain])) {
            cleanup_range(cleanup_domain, last_cleanup[cleanup_domain], new_alives[cleanup_domain]);
        }
    }
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
PTO2TensorMapEntry* TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::insert(
    const Tensor& tensor, PTO2TaskId producer_task_id, const TensorMapInsertMeta& meta) {
#if PTO2_TENSORMAP_PROFILING
    g_insert_count++;
#endif
    int32_t cleanup_domain = cleanup_domain_of(producer_task_id);
    always_assert(cleanup_domain >= 0 && cleanup_domain < NumCleanupDomains);
    if (meta.storage_domain == TensorMapStorageDomain::OWNER_MAP) {
        always_assert(meta.tensor_owner_ring < PTO2_MAX_RING_DEPTH);
    }

    uint32_t bucket_index = hash(tensor.buffer.addr);
    __builtin_prefetch(&buckets[bucket_index], 1, 0);
    int32_t local_id = lifecycle_local_of(producer_task_id);
    int32_t task_slot = local_id & (task_window_sizes[cleanup_domain] - 1);
    __builtin_prefetch(&task_entry_heads[cleanup_domain][task_slot], 1, 0);

    PTO2TensorMapEntry* entry = new_entry();
    entry->copy_from_tensor(tensor);
    entry->producer_task_id = producer_task_id;
    entry->tensor_owner_ring = meta.tensor_owner_ring;
    entry->storage_domain = meta.storage_domain;
    entry->with_alloc = meta.with_alloc;
    account_entry_added(*entry);

    entry->bucket_index = static_cast<int32_t>(bucket_index);
    entry->next_in_bucket = buckets[bucket_index];
    if (entry->next_in_bucket != nullptr) {
        entry->next_in_bucket->prev_in_bucket = entry;
    }
    buckets[bucket_index] = entry;
    entry->prev_in_bucket = nullptr;

    entry->next_in_task = task_entry_heads[cleanup_domain][task_slot];
    entry->prev_in_task = nullptr;
    if (entry->next_in_task != nullptr) {
        entry->next_in_task->prev_in_task = entry;
    }
    task_entry_heads[cleanup_domain][task_slot] = entry;
    return entry;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
void TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::remove_entry(
    PTO2TensorMapEntry& entry) {
    always_assert(entry.bucket_index != -1);
    unlink_from_task(entry);
    reclaim_entry(entry);
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
void TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::cleanup_range(
    int32_t cleanup_domain, int32_t old_last_task_alive, int32_t new_last_task_alive) {
    always_assert(cleanup_domain >= 0 && cleanup_domain < NumCleanupDomains);
    for (int32_t local_id = old_last_task_alive; local_id < new_last_task_alive; local_id++) {
        int32_t task_slot = local_id & (task_window_sizes[cleanup_domain] - 1);
        PTO2TensorMapEntry* cur_entry = task_entry_heads[cleanup_domain][task_slot];
        while (cur_entry != nullptr) {
            PTO2TensorMapEntry* next_entry = cur_entry->next_in_task;
            debug_assert(cleanup_domain_of(*cur_entry) == cleanup_domain);
            debug_assert(lifecycle_local_of(*cur_entry) == local_id);
            reclaim_entry(*cur_entry);
            cur_entry = next_entry;
        }
        task_entry_heads[cleanup_domain][task_slot] = nullptr;
    }
    last_cleanup[cleanup_domain] = new_last_task_alive;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
int32_t TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::valid_count() const {
    int32_t count = 0;
    for (int32_t i = 0; i < pool_size; i++) {
        if (entry_pool[i].bucket_index != -1 && entry_valid(entry_pool[i])) {
            count++;
        }
    }
    return count;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
bool TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::maybe_has_owner_ring(
    uint8_t owner_ring) const {
    always_assert(owner_ring < PTO2_MAX_RING_DEPTH);
    return owner_ring_entry_counts[owner_ring] > 0;
}

template <int32_t NumCleanupDomains, bool BreakOnStale>
void TensorMapShardImpl<NumCleanupDomains, BreakOnStale>::print_stats(
    const char* label) const {
    int32_t valid = 0;
    int32_t stale = 0;
    int32_t empty_buckets = 0;
    int32_t max_chain = 0;
    int64_t total_chain = 0;
    int32_t non_empty_buckets = 0;

    for (int32_t i = 0; i < pool_size; i++) {
        if (entry_pool[i].bucket_index != -1) {
            if (entry_valid(entry_pool[i])) {
                valid++;
            } else {
                stale++;
            }
        }
    }

    for (int32_t b = 0; b < num_buckets; b++) {
        int32_t chain_len = 0;
        PTO2TensorMapEntry* cur_entry = buckets[b];
        while (cur_entry != nullptr) {
            chain_len++;
            cur_entry = cur_entry->next_in_bucket;
        }
        if (chain_len == 0) {
            empty_buckets++;
        } else {
            non_empty_buckets++;
            total_chain += chain_len;
            if (chain_len > max_chain) {
                max_chain = chain_len;
            }
        }
    }

    LOG_INFO("=== TensorMapShard Statistics: %s ===", label);
    LOG_INFO("Pool size:           %d", pool_size);
    LOG_INFO("Pool next entry idx: %d", next_entry_idx);
    LOG_INFO("Pool free_num:       %d", free_num);
    LOG_INFO("Num buckets:         %d", num_buckets);
    LOG_INFO("Valid entries:       %d", valid);
    LOG_INFO("Stale entries:       %d", stale);
    LOG_INFO("Empty buckets:       %d", empty_buckets);
    LOG_INFO("Max chain len:       %d", max_chain);
    LOG_INFO("Avg chain len:       %.2f", non_empty_buckets > 0 ? static_cast<float>(total_chain) / non_empty_buckets : 0);
    for (int d = 0; d < NumCleanupDomains; d++) {
        LOG_INFO("last_task_alive[%d]: %d", d, last_task_alives[d]);
        LOG_INFO("last_cleanup[%d]:    %d", d, last_cleanup[d]);
    }
}

template struct TensorMapShardImpl<1, true>;
template struct TensorMapShardImpl<PTO2_MAX_RING_DEPTH, false>;

bool PTO2TensorMap::init(
    int32_t num_buckets, int32_t pool_size, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH]) {
    destroy();

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t owner_task_window_sizes[1] = {task_window_sizes[r]};
        if (!owner_shards[r].init(num_buckets, pool_size, owner_task_window_sizes)) {
            destroy();
            return false;
        }
    }

    if (!fallback_shard.init(num_buckets, pool_size, task_window_sizes)) {
        destroy();
        return false;
    }

    return true;
}

bool PTO2TensorMap::init_default(const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH]) {
    return init(PTO2_TENSORMAP_NUM_BUCKETS, PTO2_TENSORMAP_POOL_SIZE, task_window_sizes);
}

void PTO2TensorMap::destroy() {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        owner_shards[r].destroy();
    }
    fallback_shard.destroy();
}

void PTO2TensorMap::lookup(const Tensor& tensor, PTO2LookupResult& result) {
    always_assert(tensor_has_valid_owner_ring(tensor));
    result.count = 0;
    if (tensor.ring_id == TENSOR_RING_ID_NONE) {
        fallback_shard.refresh_lookup_domains(tensor, orch);
        fallback_shard.lookup(tensor, result);
        return;
    }
    owner_shards[tensor.ring_id].lookup(tensor, result);
    if (!fallback_shard.maybe_has_owner_ring(tensor.ring_id)) {
        return;
    }
    fallback_shard.refresh_lookup_domains(tensor, orch);
    fallback_shard.lookup(tensor, result);
}

void PTO2TensorMap::insert(
    const Tensor& tensor, PTO2TaskId producer_task_id, PTOParamType param_type, bool with_alloc) {
    always_assert(param_type == PTOParamType::OUTPUT || param_type == PTOParamType::INOUT);
    always_assert(tensor_has_valid_owner_ring(tensor));
    uint8_t producer_ring = producer_task_id.ring();
    always_assert(producer_ring < PTO2_MAX_RING_DEPTH);

    TensorMapInsertMeta meta;
    meta.tensor_owner_ring = tensor.ring_id;
    meta.with_alloc = with_alloc;

    if (tensor.ring_id == TENSOR_RING_ID_NONE) {
        meta.storage_domain = TensorMapStorageDomain::FALLBACK_MAP;
        fallback_shard.insert(tensor, producer_task_id, meta);
        return;
    }

    if (param_type == PTOParamType::OUTPUT) {
        always_assert(tensor.ring_id == producer_ring);
        meta.storage_domain = TensorMapStorageDomain::OWNER_MAP;
        owner_shards[producer_ring].insert(tensor, producer_task_id, meta);
        return;
    }

    if (tensor.ring_id == producer_ring) {
        meta.storage_domain = TensorMapStorageDomain::OWNER_MAP;
        owner_shards[producer_ring].insert(tensor, producer_task_id, meta);
    } else {
        meta.storage_domain = TensorMapStorageDomain::FALLBACK_MAP;
        fallback_shard.insert(tensor, producer_task_id, meta);
    }
}

void PTO2TensorMap::remove_entry(PTO2TensorMapEntry& entry) {
    switch (entry.storage_domain) {
        case TensorMapStorageDomain::OWNER_MAP:
            always_assert(entry.tensor_owner_ring < PTO2_MAX_RING_DEPTH);
            owner_shards[entry.storage_ring()].remove_entry(entry);
            return;
        case TensorMapStorageDomain::FALLBACK_MAP:
            fallback_shard.remove_entry(entry);
            return;
    }
    always_assert(false);
}

void PTO2TensorMap::print_stats() const {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        char label[32];
        snprintf(label, sizeof(label), "owner[%d]", r);
        owner_shards[r].print_stats(label);
    }
    fallback_shard.print_stats("fallback");
}

int32_t PTO2TensorMap::valid_count() const {
    int32_t count = fallback_shard.valid_count();
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        count += owner_shards[r].valid_count();
    }
    return count;
}

void PTO2TensorMap::sync_tensormap(uint8_t ring_id, int32_t sm_last_task_alive) {
    always_assert(ring_id < PTO2_MAX_RING_DEPTH);
    owner_shards[ring_id].sync_validity(0, sm_last_task_alive);
    if (owner_shards[ring_id].need_cleanup(0, sm_last_task_alive)) {
        owner_shards[ring_id].cleanup_range(
            0, owner_shards[ring_id].last_cleanup[0], sm_last_task_alive);
    }

    fallback_shard.sync_validity(ring_id, sm_last_task_alive);
    if (fallback_shard.need_cleanup(ring_id, sm_last_task_alive)) {
        fallback_shard.cleanup_range(
            ring_id, fallback_shard.last_cleanup[ring_id], sm_last_task_alive);
    }
}

#if PTO2_TENSORMAP_PROFILING
PTO2TensorMapProfilingData pto2_tensormap_get_profiling() {
    PTO2TensorMapProfilingData d;
    d.lookup_chain_total = g_lookup_chain_total;
    d.lookup_count = g_lookup_count;
    d.lookup_chain_max = g_lookup_chain_max;
    d.overlap_checks = g_lookup_overlap_checks;
    d.overlap_hits = g_lookup_overlap_hits;
    d.insert_count = g_insert_count;

    g_lookup_chain_total = 0;
    g_lookup_count = 0;
    g_lookup_chain_max = 0;
    g_lookup_overlap_checks = 0;
    g_lookup_overlap_hits = 0;
    g_insert_count = 0;
    return d;
}
#endif
