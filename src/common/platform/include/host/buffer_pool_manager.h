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
 * @file buffer_pool_manager.h
 * @brief Generic buffer-pool data structure shared by L2Swimlane, TensorDump,
 *        and PMU collectors. Owns:
 *
 *   - ready_queue shard(s) (mgmt → collector) with mutex/cv,
 *   - done_queue shard(s) (collector → mgmt) with mutex,
 *   - shard-local per-kind recycled-buffer pools,
 *   - dev↔host pointer mapping table,
 *   - alloc_and_register / free_buffer / resolve_host_ptr helpers.
 *
 * Owns no threads. ProfilerBase drives the mgmt loop and forwards memory
 * context here once via set_memory_context(). The Module concept contract
 * lives at the top of profiler_base.h.
 *
 * Defines the shared types used by the framework: ThreadFactory (for thread
 * creation with optional device-context binding), MemoryOps (type-erased
 * alloc/reg/free/copy callbacks), and DoneInfo (per-buffer ownership info
 * passed through done queues).
 *
 * SVM vs host-shadow (chosen at runtime by what the collector installs)
 * ---------------------------------------------------------------------
 *
 * Platforms that don't have SVM (a5: no halHostRegister) install
 * `copy_to_device` / `copy_from_device` in MemoryOps and a `reg` that
 * allocates a paired host shadow (instead of mapping a HAL view onto the
 * device pointer). The mgmt loop then:
 *   1. Calls `mirror_shm_from_device` once per tick to refresh the host
 *      shadow, then writes back only the fields it actually modifies via
 *      narrow `write_range_to_device(field_ptr, sizeof(field))` calls.
 *      A bulk host→device write-back is deliberately avoided — it would
 *      race with AICPU writes to device-only fields (current_buf_ptr,
 *      total/dropped/mismatch counters, queue_tails, free_queue.head, and
 *      on a5 L2SwimlaneAicpuPhaseHeader::magic).
 *   2. Pulls each popped buffer's contents from device via
 *      `copy_buffer_from_device` inside ProfilerAlgorithms::process_entry
 *      before delivering it to the collector.
 *
 * `release_owned_buffers` frees both the device pointer (via `release_fn`)
 * and any paired host shadow that the framework itself malloc'd. Ownership
 * is tracked explicitly in `malloc_shadows_`: only shadows allocated via
 * `default_host_shadow_register` or the `copy_to_device_` branch of
 * `ProfilerBase::alloc_paired_buffer` are added to the set, so HAL-managed
 * mappings (e.g. `halHostRegister` results on a2a3 onboard) never see a
 * spurious `std::free`. The earlier `host_ptr != dev_ptr` alias check is
 * not sufficient on its own — `halHostRegister` returns a host VA that
 * may or may not coincide with the device VA, and feeding either case to
 * `std::free` is UB.
 *
 * Platforms with SVM (a2a3: halHostRegister maps device pointers into host
 * address space) leave `copy_to_device` / `copy_from_device` null and pass
 * the same pointer as both shm_dev and shm_host (or shm_size=0). All
 * mirror_* / write_range_* / read_range_* / copy_buffer_* methods then
 * short-circuit through the internal `if (!ops_.copy_to_device) return 0;`
 * guard and cost a single function call per tick (zero memcpy work on the
 * SVM path).
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_BUFFER_POOL_MANAGER_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_BUFFER_POOL_MANAGER_H_

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/unified_log.h"

namespace profiling_common {

/**
 * Thread factory for spawning the mgmt thread with optional device-context
 * binding. Pass `create_thread()` from device_runner to get a sim/onboard
 * device-bound worker; pass {} (default) to fall back to a bare std::thread.
 */
using ThreadFactory = std::function<std::thread(std::function<void()>)>;

/**
 * Type-erased memory-op callbacks used by BufferPoolManager.
 *
 * - alloc:            allocate `size` bytes of device memory; return nullptr
 *                     on failure.
 * - reg:              "register" dev_ptr for host visibility. On a5 this
 *                     allocates a paired host shadow (malloc + memset 0 +
 *                     copy_to_device of the zeros) and writes its address to
 *                     *host_ptr_out. ProfilerBase::start always installs a
 *                     non-null reg wrapper — collectors do not need to
 *                     branch.
 * - free_:            free a previously allocated device pointer.
 * - copy_to_device:   memcpy host → device. Used at init/teardown for the
 *                     bulk shm push, and used by the mgmt loop via
 *                     `write_range_to_device` to push narrow host-modified
 *                     fields back (advanced `queue_heads[q]`, refilled
 *                     `free_queue.tail` + `buffer_ptrs[slot]`) without
 *                     clobbering AICPU-owned fields.
 * - copy_from_device: memcpy device → host, used at the top of every mgmt
 *                     tick to mirror device-side `queue_tails` /
 *                     BufferState updates into the host shadow.
 */
struct MemoryOps {
    std::function<void *(size_t)> alloc;
    std::function<int(void *dev_ptr, size_t size, int device_id, void **host_ptr_out)> reg;
    std::function<int(void *dev_ptr)> free_;
    std::function<int(void *dev_dst, const void *host_src, size_t size)> copy_to_device;
    std::function<int(void *host_dst, const void *dev_src, size_t size)> copy_from_device;
};

/**
 * Per-buffer ownership info threaded through a done queue shard so that the
 * mgmt thread, when it recycles a finished buffer, knows which per-kind pool it
 * came from.
 */
struct DoneInfo {
    void *dev_ptr;
    int kind;  // [0, Module::kBufferKinds)
};

template <typename Module, typename = void>
struct ProfilerModuleCollectorThreadCount {
    static constexpr int value = 1;
};

template <typename Module>
struct ProfilerModuleCollectorThreadCount<Module, std::void_t<decltype(Module::kCollectorThreadCount)>> {
    static constexpr int value = Module::kCollectorThreadCount;
};

template <typename Module>
class BufferPoolManager {
    // Static checks for the Module concept. Required type aliases trigger
    // clear "no type named X in Module" errors at instantiation if missing;
    // the explicit static_asserts cover constants and surface invariants.
    using _DataHeaderRequired = typename Module::DataHeader;
    using _ReadyEntryRequired = typename Module::ReadyEntry;
    using _ReadyBufferInfoRequired = typename Module::ReadyBufferInfo;
    static_assert(Module::kBufferKinds > 0, "Module::kBufferKinds must be > 0");

public:
    using ReadyBufferInfo = typename Module::ReadyBufferInfo;
    static constexpr int kCollectorShardCount = ProfilerModuleCollectorThreadCount<Module>::value;
    static_assert(kCollectorShardCount >= 1, "Module::kCollectorThreadCount must be >= 1");

    BufferPoolManager() :
        ready_shards_(kCollectorShardCount),
        done_shards_(kCollectorShardCount) {}
    ~BufferPoolManager() = default;

    BufferPoolManager(const BufferPoolManager &) = delete;
    BufferPoolManager &operator=(const BufferPoolManager &) = delete;

    /**
     * Configure the buffer pool's memory context. Called by ProfilerBase::start()
     * before any allocator-touching method (alloc_and_register / free_buffer /
     * resolve_host_ptr / drain_done_into_recycled triggered by the mgmt loop)
     * is invoked. Must NOT be called concurrently with the mgmt thread.
     *
     * @param ops              Memory-op callbacks (alloc/reg/free/copy_*).
     * @param shared_mem_dev   Device base of the subsystem's shared memory.
     * @param shared_mem_host  Host shadow of the same region.
     * @param shm_size         Total bytes of the shared-memory region (used
     *                         by the mgmt loop's per-tick mirror).
     * @param device_id        Forwarded to ops.reg.
     */
    void
    set_memory_context(MemoryOps ops, void *shared_mem_dev, void *shared_mem_host, size_t shm_size, int device_id) {
        ops_ = std::move(ops);
        shared_mem_dev_ = shared_mem_dev;
        shared_mem_host_ = shared_mem_host;
        shm_size_ = shm_size;
        device_id_ = device_id;
    }

    /**
     * Release every device buffer the framework currently owns: recycled
     * pools, done queues, and ready queues. Buffers still in the per-pool
     * free_queue or held as current_buf_ptr are NOT touched — those belong
     * to the collector and must be released by it (the AICPU may still be
     * referencing them via shared memory until execution ends).
     *
     * For each unique device pointer freed, the paired host shadow is
     * `std::free`d ONLY if it lives in `malloc_shadows_` (i.e. the
     * framework itself malloc'd it via `default_host_shadow_register` or
     * `ProfilerBase::alloc_paired_buffer`'s copy-to-device branch). HAL
     * mappings (e.g. `halHostRegister` results) are never freed here.
     *
     * `release_fn(dev_ptr)` is invoked once per unique pointer; the
     * collector is expected to call its free_cb on the device pointer.
     *
     * Only safe to call after ProfilerBase::stop() has joined the mgmt thread.
     */
    template <typename ReleaseFn>
    void release_owned_buffers(const ReleaseFn &release_fn) {
        std::unordered_map<void *, bool> seen;
        auto release_once = [&](void *p) {
            if (p == nullptr) return;
            if (seen.emplace(p, true).second) {
                auto it = dev_to_host_.find(p);
                void *host_ptr = (it != dev_to_host_.end()) ? it->second : nullptr;
                release_fn(p);
                if (host_ptr != nullptr && malloc_shadows_.erase(host_ptr) > 0) {
                    std::free(host_ptr);
                }
                if (it != dev_to_host_.end()) {
                    dev_to_host_.erase(it);
                }
            }
        };

        for (auto &shard_pools : recycled_) {
            for (auto &pool : shard_pools) {
                for (void *p : pool)
                    release_once(p);
                pool.clear();
            }
        }
        {
            for (auto &shard : done_shards_) {
                std::scoped_lock<std::mutex> lock(shard.mutex);
                while (!shard.queue.empty()) {
                    release_once(shard.queue.front().dev_ptr);
                    shard.queue.pop();
                }
            }
        }
        {
            for (auto &shard : ready_shards_) {
                std::scoped_lock<std::mutex> lock(shard.mutex);
                while (!shard.queue.empty()) {
                    release_once(shard.queue.front().dev_buffer_ptr);
                    shard.queue.pop();
                }
            }
        }
    }

    /**
     * Drop the dev↔host mapping table — call after the collector has freed
     * its share of buffers (free_queue + current_buf_ptr) and there are no
     * further resolve_host_ptr() lookups expected. `std::free`s any host
     * shadow still listed in `malloc_shadows_` (collectors may have invoked
     * free_cb on the dev pointer without going through release_owned_buffers).
     * HAL mappings are not touched.
     */
    void clear_mappings() {
        for (auto &kv : dev_to_host_) {
            if (kv.second != nullptr && malloc_shadows_.count(kv.second) > 0) {
                std::free(kv.second);
            }
        }
        dev_to_host_.clear();
        malloc_shadows_.clear();
    }

    /**
     * Abort-path cleanup: free EVERY framework-tracked device pointer (via
     * `release_fn`) and every framework-malloc'd host shadow, then clear all
     * containers. Distinct from `release_owned_buffers()` + `clear_mappings()`
     * because this also catches buffers parked in callers' SPSC free_queues
     * (which the framework tracked via `register_mapping` but does not own a
     * queue for). Intended for `init()` error paths where `finalize()` has
     * not run.
     *
     * Drains recycled/done/ready first (just discards — release goes via
     * dev_to_host_ to avoid double-free) and then iterates the full
     * dev→host map. Each unique dev_ptr is released exactly once.
     */
    template <typename ReleaseFn>
    void release_all_owned(const ReleaseFn &release_fn) {
        for (auto &shard_pools : recycled_) {
            for (auto &pool : shard_pools)
                pool.clear();
        }
        for (auto &shard : done_shards_) {
            std::scoped_lock<std::mutex> lock(shard.mutex);
            std::queue<DoneInfo>().swap(shard.queue);
        }
        for (auto &shard : ready_shards_) {
            std::scoped_lock<std::mutex> lock(shard.mutex);
            std::queue<ReadyBufferInfo>().swap(shard.queue);
        }
        for (auto &kv : dev_to_host_) {
            if (kv.first != nullptr) {
                release_fn(kv.first);
            }
            // erase-based check (matches release_owned_buffers): atomic
            // check-and-remove guards against a double-free if any duplicate
            // mapping ever sneaks into dev_to_host_.
            if (kv.second != nullptr && malloc_shadows_.erase(kv.second) > 0) {
                std::free(kv.second);
            }
        }
        dev_to_host_.clear();
        malloc_shadows_.clear();
    }

    // -------------------------------------------------------------------------
    // Per-tick mirror of the shared-memory region
    // -------------------------------------------------------------------------

    /**
     * Pull the entire device-side shared-memory region into the host shadow.
     * Called at the top of every mgmt tick so that subsequent reads of
     * `queue_tails`, `BufferState::current_buf_ptr`, etc. see fresh values.
     */
    int mirror_shm_from_device() {
        if (shared_mem_host_ == nullptr || shared_mem_dev_ == nullptr || shm_size_ == 0) {
            return 0;
        }
        if (!ops_.copy_from_device) return 0;
        return ops_.copy_from_device(shared_mem_host_, shared_mem_dev_, shm_size_);
    }

    /**
     * Push a single field/range from host shadow to its mirrored device
     * location. `host_field_ptr` must lie inside the host shm shadow
     * (`[shared_mem_host_, shared_mem_host_ + shm_size_)`). Used by the
     * mgmt loop to avoid bulk writing the entire shm region, which would
     * clobber device-only counters and current_buf_ptr values written by
     * AICPU between the from/to mirror calls.
     *
     * Accepts `const volatile void*` so callers can pass the address of
     * volatile fields (queue_heads[], free_queue.tail, free_queue.buffer_ptrs[])
     * without an explicit cast at the call site.
     *
     * Returns the underlying ops result, or -1 on bounds violation.
     */
    int write_range_to_device(const volatile void *host_field_ptr, size_t size) {
        if (shared_mem_host_ == nullptr || shared_mem_dev_ == nullptr || shm_size_ == 0) {
            return 0;
        }
        if (!ops_.copy_to_device) return 0;
        const auto *host_base = static_cast<const char *>(shared_mem_host_);
        const auto *host_field = const_cast<const char *>(static_cast<const volatile char *>(host_field_ptr));
        if (host_field < host_base || host_field + size > host_base + shm_size_) {
            LOG_ERROR(
                "BufferPoolManager::write_range_to_device: field [%p, %p) outside shm [%p, %p)",
                static_cast<const void *>(host_field), static_cast<const void *>(host_field + size),
                static_cast<const void *>(host_base), static_cast<const void *>(host_base + shm_size_)
            );
            return -1;
        }
        size_t offset = static_cast<size_t>(host_field - host_base);
        void *dev_field = static_cast<char *>(shared_mem_dev_) + offset;
        return ops_.copy_to_device(dev_field, host_field, size);
    }

    /**
     * Re-pull a single field/range from device into the host shadow.
     * Symmetric counterpart of `write_range_to_device`. Used to refresh
     * a specific field after the per-tick `mirror_shm_from_device` to
     * defeat a torn-read race: the bulk mirror is not atomic w.r.t.
     * concurrent AICPU writes, so a producer-published entry (e.g. a
     * `queues[t][tail]` slot) may be observed half-written if it was
     * mirrored before AICPU finished writing it. Re-reading the entry
     * after observing `head < tail` gives the latest device-side bytes.
     *
     * Accepts `volatile void*` so callers can pass the address of volatile
     * fields without an explicit cast.
     *
     * Returns the underlying ops result, or -1 on bounds violation.
     */
    int read_range_from_device(volatile void *host_field_ptr, size_t size) {
        if (shared_mem_host_ == nullptr || shared_mem_dev_ == nullptr || shm_size_ == 0) {
            return 0;
        }
        if (!ops_.copy_from_device) return 0;
        const auto *host_base = static_cast<const char *>(shared_mem_host_);
        const auto *host_field = const_cast<const char *>(static_cast<volatile char *>(host_field_ptr));
        if (host_field < host_base || host_field + size > host_base + shm_size_) {
            LOG_ERROR(
                "BufferPoolManager::read_range_from_device: field [%p, %p) outside shm [%p, %p)",
                static_cast<const void *>(host_field), static_cast<const void *>(host_field + size),
                static_cast<const void *>(host_base), static_cast<const void *>(host_base + shm_size_)
            );
            return -1;
        }
        size_t offset = static_cast<size_t>(host_field - host_base);
        const void *dev_field = static_cast<const char *>(shared_mem_dev_) + offset;
        return ops_.copy_from_device(const_cast<void *>(static_cast<const void *>(host_field)), dev_field, size);
    }

    /**
     * Pull a single buffer's contents (e.g. an L2SwimlaneAicpuTaskBuffer / PmuBuffer /
     * DumpMetaBuffer) from device to its host shadow. Called by
     * ProfilerAlgorithms::process_entry after resolving the host pointer
     * for a popped ready entry, before delivering it to the collector.
     */
    int copy_buffer_from_device(void *host_dst, void *dev_src, size_t size) {
        if (!ops_.copy_from_device) return 0;
        return ops_.copy_from_device(host_dst, dev_src, size);
    }

    /**
     * Push a single buffer's contents from host shadow to device. Currently
     * unused by the mgmt loop (AICPU resets buffer state itself when it
     * pops from free_queue), but exposed for collector-side use cases.
     */
    int copy_buffer_to_device(void *dev_dst, const void *host_src, size_t size) {
        if (!ops_.copy_to_device) return 0;
        return ops_.copy_to_device(dev_dst, host_src, size);
    }

    // -------------------------------------------------------------------------
    // ready_queue shards: mgmt threads push, collector threads pop
    // -------------------------------------------------------------------------

    void push_to_ready(const ReadyBufferInfo &info, int shard_index = 0) {
        auto &shard = ready_shards_[normalize_shard(shard_index)];
        {
            std::scoped_lock<std::mutex> lock(shard.mutex);
            shard.queue.push(info);
        }
        shard.cv.notify_one();
    }

    bool try_pop_ready(ReadyBufferInfo &out, int shard_index = 0) {
        auto &shard = ready_shards_[normalize_shard(shard_index)];
        std::scoped_lock<std::mutex> lock(shard.mutex);
        if (shard.queue.empty()) return false;
        out = shard.queue.front();
        shard.queue.pop();
        return true;
    }

    bool wait_pop_ready(ReadyBufferInfo &out, std::chrono::milliseconds timeout, int shard_index = 0) {
        auto &shard = ready_shards_[normalize_shard(shard_index)];
        std::unique_lock<std::mutex> lock(shard.mutex);
        if (!shard.cv.wait_for(lock, timeout, [&shard] {
                return !shard.queue.empty();
            })) {
            return false;
        }
        out = shard.queue.front();
        shard.queue.pop();
        return true;
    }

    // -------------------------------------------------------------------------
    // done_queue shards: collector threads report buffers they have finished
    // copying; mgmt folds them back into the same shard's recycled pool of the
    // right kind.
    // -------------------------------------------------------------------------

    void notify_copy_done(void *dev_ptr, int kind, int shard_index = 0) {
        auto &shard = done_shards_[normalize_shard(shard_index)];
        std::scoped_lock<std::mutex> lock(shard.mutex);
        shard.queue.push(DoneInfo{dev_ptr, kind});
    }

    // -------------------------------------------------------------------------
    // Helpers used from Module::process_entry / proactive_replenish
    // -------------------------------------------------------------------------

    /**
     * Allocate a new device buffer and pair it with a host shadow via
     * ops_.reg. Tracks the resulting dev→host mapping so resolve_host_ptr()
     * can find it on subsequent ready-queue pops.
     *
     * @param size              Byte size to allocate.
     * @param[out] host_ptr_out Host shadow pointer.
     * @return                  Device pointer, or nullptr on failure.
     */
    void *alloc_and_register(size_t size, void **host_ptr_out) {
        void *dev_ptr = ops_.alloc(size);
        if (dev_ptr == nullptr) {
            *host_ptr_out = nullptr;
            return nullptr;
        }
        void *host_ptr = nullptr;
        int rc = ops_.reg(dev_ptr, size, device_id_, &host_ptr);
        if (rc != 0 || host_ptr == nullptr) {
            LOG_ERROR("BufferPoolManager: register failed: %d", rc);
            // Best-effort dev free; no shadow was registered yet.
            if (ops_.free_) {
                ops_.free_(dev_ptr);
            }
            *host_ptr_out = nullptr;
            return nullptr;
        }
        *host_ptr_out = host_ptr;
        {
            std::scoped_lock<std::mutex> lock(mapping_mutex_);
            dev_to_host_[dev_ptr] = host_ptr;
        }
        return dev_ptr;
    }

    /**
     * Free a device pointer + paired host shadow tracked in dev_to_host_.
     * Currently unused by the mgmt loop (recycle path keeps buffers alive)
     * but kept for symmetry with a2a3.
     */
    void free_buffer(void *dev_ptr) {
        if (dev_ptr == nullptr) return;
        void *host_ptr = nullptr;
        bool free_host_shadow = false;
        {
            std::scoped_lock<std::mutex> lock(mapping_mutex_);
            auto it = dev_to_host_.find(dev_ptr);
            host_ptr = (it != dev_to_host_.end()) ? it->second : nullptr;
            if (it != dev_to_host_.end()) {
                dev_to_host_.erase(it);
            }
            free_host_shadow = (host_ptr != nullptr && malloc_shadows_.erase(host_ptr) > 0);
        }
        if (ops_.free_) {
            ops_.free_(dev_ptr);
        }
        if (free_host_shadow) {
            std::free(host_ptr);
        }
    }

    /**
     * Resolve a device pointer to the host-mapped pointer recorded at
     * alloc_and_register / register_mapping time.
     */
    void *resolve_host_ptr(void *dev_ptr) {
        std::scoped_lock<std::mutex> lock(mapping_mutex_);
        auto it = dev_to_host_.find(dev_ptr);
        if (it != dev_to_host_.end()) return it->second;
        LOG_ERROR("BufferPoolManager: no host mapping for dev_ptr=%p", dev_ptr);
        return nullptr;
    }

    /**
     * Register an externally-allocated mapping. Used by the Collector during
     * initialize() when it pre-allocates buffers and wants the mgmt thread
     * to be able to resolve them later.
     */
    void register_mapping(void *dev_ptr, void *host_ptr) {
        std::scoped_lock<std::mutex> lock(mapping_mutex_);
        dev_to_host_[dev_ptr] = host_ptr;
    }

    /**
     * Claim ownership of a host shadow that the framework malloc'd. Only
     * shadows tracked here are `std::free`d by `clear_mappings()`,
     * `release_owned_buffers()`, and `free_buffer()` — HAL-managed
     * mappings (e.g. `halHostRegister` results) must NOT be added here.
     */
    void add_malloc_shadow(void *host_ptr) {
        if (host_ptr != nullptr) {
            std::scoped_lock<std::mutex> lock(mapping_mutex_);
            malloc_shadows_.insert(host_ptr);
        }
    }

    /**
     * Pull from the recycled pool of the given kind, or return nullptr if
     * empty. Caller is responsible for resolving host_ptr (via
     * resolve_host_ptr) before handing the buffer back to AICPU.
     */
    void *pop_recycled(int kind, int shard_index = 0) {
        auto shard = normalize_shard(shard_index);
        std::scoped_lock<std::mutex> lock(recycled_mutexes_[shard][kind]);
        auto &pool = recycled_[shard][kind];
        if (pool.empty()) return nullptr;
        void *p = pool.back();
        pool.pop_back();
        return p;
    }

    void *pop_recycled_any(int kind, int preferred_shard = 0) {
        if (void *p = pop_recycled(kind, preferred_shard); p != nullptr) return p;
        const auto preferred = normalize_shard(preferred_shard);
        for (size_t s = 0; s < recycled_.size(); s++) {
            if (s == preferred) continue;
            if (void *p = pop_recycled(kind, static_cast<int>(s)); p != nullptr) return p;
        }
        return nullptr;
    }

    void push_recycled(int kind, void *dev_ptr, int shard_index = 0) {
        auto shard = normalize_shard(shard_index);
        std::scoped_lock<std::mutex> lock(recycled_mutexes_[shard][kind]);
        recycled_[shard][kind].push_back(dev_ptr);
    }

    size_t recycled_count(int kind) const {
        size_t total = 0;
        for (size_t shard = 0; shard < recycled_.size(); shard++) {
            std::scoped_lock<std::mutex> lock(recycled_mutexes_[shard][kind]);
            total += recycled_[shard][kind].size();
        }
        return total;
    }

    bool recycled_empty() const {
        for (size_t shard = 0; shard < recycled_.size(); shard++) {
            for (int kind = 0; kind < Module::kBufferKinds; kind++) {
                std::scoped_lock<std::mutex> lock(recycled_mutexes_[shard][kind]);
                if (!recycled_[shard][kind].empty()) return false;
            }
        }
        return true;
    }

    template <typename Fn>
    decltype(auto) with_free_queue_writer(const void *queue_key, Fn &&fn) {
        std::scoped_lock<std::mutex> lock(free_queue_mutexes_[free_queue_lock_index(queue_key)]);
        return fn();
    }

    /**
     * Drain everything currently in done queue shards back into the per-kind
     * recycled pool. May be called from Module::process_entry when its
     * primary recycled pool ran out, to harvest buffers the collector freed
     * in the meantime.
     */
    size_t drain_done_into_recycled(int shard_index) {
        auto &shard = done_shards_[normalize_shard(shard_index)];
        size_t drained = 0;
        std::scoped_lock<std::mutex> lock(shard.mutex);
        while (!shard.queue.empty()) {
            const DoneInfo &info = shard.queue.front();
            push_recycled(info.kind, info.dev_ptr, shard_index);
            shard.queue.pop();
            drained++;
        }
        return drained;
    }

    size_t drain_done_into_recycled() {
        size_t drained = 0;
        for (size_t shard = 0; shard < done_shards_.size(); shard++) {
            drained += drain_done_into_recycled(static_cast<int>(shard));
        }
        return drained;
    }

    void *shared_mem_dev() const { return shared_mem_dev_; }
    void *shared_mem_host() const { return shared_mem_host_; }
    int device_id() const { return device_id_; }

private:
    struct ReadyQueueShard {
        std::mutex mutex;
        std::condition_variable cv;
        std::queue<ReadyBufferInfo> queue;
    };

    struct DoneQueueShard {
        std::mutex mutex;
        std::queue<DoneInfo> queue;
    };

    static size_t normalize_shard(int shard_index) {
        if (shard_index < 0) return 0;
        return static_cast<size_t>(shard_index) % static_cast<size_t>(kCollectorShardCount);
    }

    // Subsystem inputs (set by ProfilerBase::start via set_memory_context).
    void *shared_mem_dev_{nullptr};
    void *shared_mem_host_{nullptr};
    size_t shm_size_{0};
    int device_id_{-1};
    MemoryOps ops_;

    // mgmt → collector
    std::vector<ReadyQueueShard> ready_shards_;

    // collector → mgmt
    std::vector<DoneQueueShard> done_shards_;

    // Host-side pointer mappings are shared across all collector shards.
    mutable std::mutex mapping_mutex_;
    static constexpr size_t kFreeQueueLockStripes = 64;

    static size_t free_queue_lock_index(const void *queue_key) {
        auto raw = reinterpret_cast<uintptr_t>(queue_key);
        return (raw >> 6) % kFreeQueueLockStripes;
    }

    std::array<std::mutex, kFreeQueueLockStripes> free_queue_mutexes_;

    // dev → host mapping (single source of truth for resolve_host_ptr)
    std::unordered_map<void *, void *> dev_to_host_;

    // Host shadows the framework itself malloc'd (via
    // `default_host_shadow_register` or `ProfilerBase::alloc_paired_buffer`'s
    // copy-to-device branch). Only these are `std::free`d on teardown —
    // HAL-managed mappings (halHostRegister) live outside this set.
    std::unordered_set<void *> malloc_shadows_;

    // Local recycled buffer pools indexed by collector shard, then Module-defined kind id.
    std::array<std::array<std::vector<void *>, Module::kBufferKinds>, kCollectorShardCount> recycled_;
    mutable std::array<std::array<std::mutex, Module::kBufferKinds>, kCollectorShardCount> recycled_mutexes_;
};

}  // namespace profiling_common

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_BUFFER_POOL_MANAGER_H_
