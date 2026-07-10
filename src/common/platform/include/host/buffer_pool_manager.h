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
 * @brief Generic buffer-pool data structure shared by L2Swimlane, PMU,
 *        DepGen, TensorDump, and ScopeStats collectors. Owns:
 *
 *   - ready_queue shard(s) (mgmt → collector) as SPSC rings,
 *   - done_queue shard(s) (collector → mgmt) as SPSC rings,
 *   - shard-local per-kind recycled-buffer SPSC rings,
 *   - mutex-protected retired-buffer pools for exceptional teardown paths,
 *   - dev↔host pointer mapping table with block-range resolution,
 *   - batched block allocation / free_buffer / resolve_host_ptr helpers.
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
 * `release_owned_buffers` releases device allocations via the collector's
 * `release_fn`, canonicalizing carved sub-buffers back to their registered
 * block base first. Paired host shadows that the framework malloc'd are
 * released later by `clear_mappings()` or immediately by `release_all_owned()`.
 * Ownership is tracked explicitly in `malloc_shadows_`: only shadows allocated
 * via `default_host_shadow_register` or the `copy_to_device_` branch of
 * `ProfilerBase::alloc_paired_buffer` are added to the set, so HAL-managed
 * mappings (e.g. `halHostRegister` results on a2a3 onboard) never see a
 * spurious `std::free`. The earlier `host_ptr != dev_ptr` alias check is not
 * sufficient on its own — `halHostRegister` returns a host VA that may or may
 * not coincide with the device VA, and feeding either case to `std::free` is UB.
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

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
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

template <typename Module, typename = void>
struct ProfilerModuleHostQueueCapacity {
    static constexpr size_t value = 1024;
};

template <typename Module>
struct ProfilerModuleHostQueueCapacity<Module, std::void_t<decltype(Module::kReadyQueueSize)>> {
    static constexpr size_t value = Module::kReadyQueueSize > 0 ? Module::kReadyQueueSize : 1;
};

template <typename Module, typename = void>
struct ProfilerModuleHostPoolQueueCapacity {
    static constexpr size_t value = 1024;
};

template <typename Module>
struct ProfilerModuleHostPoolQueueCapacity<Module, std::void_t<decltype(Module::kHostPoolQueueSize)>> {
    static constexpr size_t value = Module::kHostPoolQueueSize > 0 ? Module::kHostPoolQueueSize : 1;
};

template <typename Module, typename = void>
struct ProfilerModuleHostRecycledQueueCapacity {
    static constexpr size_t value = ProfilerModuleHostPoolQueueCapacity<Module>::value;
};

template <typename Module>
struct ProfilerModuleHostRecycledQueueCapacity<Module, std::void_t<decltype(Module::kHostRecycledQueueSize)>> {
    static constexpr size_t value = Module::kHostRecycledQueueSize > 0 ? Module::kHostRecycledQueueSize : 1;
};

template <typename T, size_t Capacity>
class SpscRing {
    static_assert(Capacity > 0, "SpscRing capacity must be > 0");

public:
    bool push(const T &item) {
        uint64_t head = head_.load(std::memory_order_relaxed);
        uint64_t tail = tail_.load(std::memory_order_acquire);
        if (head - tail >= Capacity) {
            return false;
        }
        entries_[head % Capacity] = item;
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    bool pop(T &out) {
        uint64_t tail = tail_.load(std::memory_order_relaxed);
        uint64_t head = head_.load(std::memory_order_acquire);
        if (tail == head) {
            return false;
        }
        out = entries_[tail % Capacity];
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    bool empty() const {
        uint64_t tail = tail_.load(std::memory_order_acquire);
        uint64_t head = head_.load(std::memory_order_acquire);
        return tail == head;
    }

    size_t size() const {
        uint64_t tail = tail_.load(std::memory_order_acquire);
        uint64_t head = head_.load(std::memory_order_acquire);
        uint64_t count = head - tail;
        return count > Capacity ? Capacity : static_cast<size_t>(count);
    }

    void clear() {
        T ignored{};
        while (pop(ignored)) {}
    }

private:
    std::array<T, Capacity> entries_{};
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) std::atomic<uint64_t> tail_{0};
};

class RetiredPool {
public:
    bool push(void *item) {
        std::scoped_lock<std::mutex> lock(mutex_);
        try {
            entries_.push_back(item);
        } catch (...) {
            return false;
        }
        return true;
    }

    bool pop(void *&out) {
        std::scoped_lock<std::mutex> lock(mutex_);
        if (entries_.empty()) return false;
        out = entries_.back();
        entries_.pop_back();
        return true;
    }

    void clear() {
        std::scoped_lock<std::mutex> lock(mutex_);
        entries_.clear();
    }

    bool empty() const {
        std::scoped_lock<std::mutex> lock(mutex_);
        return entries_.empty();
    }

    size_t size() const {
        std::scoped_lock<std::mutex> lock(mutex_);
        return entries_.size();
    }

private:
    mutable std::mutex mutex_;
    std::vector<void *> entries_;
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
    static constexpr size_t kReadyQueueCapacity = ProfilerModuleHostQueueCapacity<Module>::value;
    static constexpr size_t kPoolQueueCapacity = ProfilerModuleHostPoolQueueCapacity<Module>::value;
    static constexpr size_t kRecycledQueueCapacity = ProfilerModuleHostRecycledQueueCapacity<Module>::value;
    static constexpr size_t kHostQueueCapacity = kReadyQueueCapacity;

    BufferPoolManager() = default;
    ~BufferPoolManager() = default;

    BufferPoolManager(const BufferPoolManager &) = delete;
    BufferPoolManager &operator=(const BufferPoolManager &) = delete;

    /**
     * Configure the buffer pool's memory context. Called by ProfilerBase::start()
     * before any allocator-touching method (alloc_and_register_block /
     * free_buffer / resolve_host_ptr / drain_done_into_recycled) is invoked.
     * Must NOT be called concurrently with the mgmt thread.
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
     * `release_fn(dev_ptr)` is invoked once per unique allocation base.
     * Carved sub-buffers from the same registered block are canonicalized
     * back to that block base before invoking release_fn.
     *
     * Only safe to call after ProfilerBase::stop() has joined the mgmt thread.
     */
    template <typename ReleaseFn>
    void release_owned_buffers(const ReleaseFn &release_fn) {
        std::unordered_set<void *> seen;
        auto release_once = [&](void *p) {
            if (p == nullptr) return;
            void *release_ptr = release_pointer_for(p);
            if (release_ptr != nullptr && seen.insert(release_ptr).second) {
                release_fn(release_ptr);
            }
        };

        for (auto &shard_pools : recycled_) {
            for (auto &pool : shard_pools) {
                void *p = nullptr;
                while (pool.pop(p)) {
                    release_once(p);
                }
            }
        }
        for (auto &shard_pools : retired_) {
            for (auto &pool : shard_pools) {
                void *p = nullptr;
                while (pool.pop(p)) {
                    release_once(p);
                }
            }
        }
        for (auto &shard : done_shards_) {
            DoneInfo info{};
            while (shard.queue.pop(info)) {
                release_once(info.dev_ptr);
            }
        }
        for (auto &shard : ready_shards_) {
            ReadyBufferInfo info{};
            while (shard.queue.pop(info)) {
                release_once(info.dev_buffer_ptr);
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
            if (kv.second != nullptr && malloc_shadows_.erase(kv.second) > 0) {
                std::free(kv.second);
            }
        }
        dev_to_host_.clear();
        block_ranges_.clear();
        released_allocations_.clear();
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
     * dev_to_host_ to avoid double-free) and then iterates the full mapping
     * table. Each unique allocation base is released exactly once.
     */
    template <typename ReleaseFn>
    void release_all_owned(const ReleaseFn &release_fn) {
        for (auto &shard_pools : recycled_) {
            for (auto &pool : shard_pools)
                pool.clear();
        }
        for (auto &shard_pools : retired_) {
            for (auto &pool : shard_pools)
                pool.clear();
        }
        for (auto &shard : done_shards_) {
            shard.queue.clear();
        }
        for (auto &shard : ready_shards_) {
            shard.queue.clear();
        }
        std::unordered_set<void *> release_ptrs;
        for (const auto &kv : dev_to_host_) {
            if (kv.first != nullptr) {
                if (void *release_ptr = allocation_base_for_locked(kv.first); release_ptr != nullptr) {
                    release_ptrs.insert(release_ptr);
                }
            }
        }
        for (void *p : release_ptrs) {
            release_fn(p);
        }
        for (void *host_ptr : malloc_shadows_) {
            std::free(host_ptr);
        }
        dev_to_host_.clear();
        block_ranges_.clear();
        released_allocations_.clear();
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

    bool push_to_ready(const ReadyBufferInfo &info, int shard_index = 0) {
        auto &shard = ready_shards_[normalize_shard(shard_index)];
        if (!shard.queue.push(info)) {
            LOG_ERROR("BufferPoolManager: ready queue full for shard=%d capacity=%zu", shard_index, kHostQueueCapacity);
            return false;
        }
        shard.notify_epoch.fetch_add(1, std::memory_order_release);
        shard.cv.notify_one();
        return true;
    }

    bool try_pop_ready(ReadyBufferInfo &out, int shard_index = 0) {
        auto &shard = ready_shards_[normalize_shard(shard_index)];
        return shard.queue.pop(out);
    }

    bool wait_pop_ready(ReadyBufferInfo &out, std::chrono::milliseconds timeout, int shard_index = 0) {
        auto &shard = ready_shards_[normalize_shard(shard_index)];
        if (shard.queue.pop(out)) return true;

        uint64_t seen = shard.notify_epoch.load(std::memory_order_acquire);
        std::unique_lock<std::mutex> lock(shard.wait_mutex);
        if (!shard.cv.wait_for(lock, timeout, [&shard, seen] {
                return shard.notify_epoch.load(std::memory_order_acquire) != seen || !shard.queue.empty();
            })) {
            return false;
        }
        return shard.queue.pop(out);
    }

    // -------------------------------------------------------------------------
    // done_queue shards: collector threads report buffers they have finished
    // copying; mgmt folds them back into the same shard's recycled pool of the
    // right kind.
    // -------------------------------------------------------------------------

    bool notify_copy_done(void *dev_ptr, int kind, int shard_index = 0) {
        auto &shard = done_shards_[normalize_shard(shard_index)];
        if (!shard.queue.push(DoneInfo{dev_ptr, kind})) {
            LOG_ERROR(
                "BufferPoolManager: done queue full for shard=%d kind=%d capacity=%zu", shard_index, kind,
                kPoolQueueCapacity
            );
            return false;
        }
        return true;
    }

    bool try_pop_done(DoneInfo &out, int shard_index = 0) {
        auto &shard = done_shards_[normalize_shard(shard_index)];
        return shard.queue.pop(out);
    }

    // -------------------------------------------------------------------------
    // Helpers used from Module::process_entry / proactive_replenish
    // -------------------------------------------------------------------------

    /**
     * Allocate one device block, register it once, carve it into fixed-size
     * buffers, then publish the carved buffer starts to one or more recycled
     * lanes. The block mapping lets resolve_host_ptr() translate any carved
     * device pointer by range offset, so the HAL registration cost is paid once
     * per batch rather than once per buffer.
     */
    size_t allocate_recycled_batch(int kind, size_t buffer_size, int count, int shard_index = -1) {
        if (count <= 0 || buffer_size == 0) return 0;

        constexpr size_t kCarveAlignment = 64;
        size_t stride = align_up(buffer_size, kCarveAlignment);
        if (stride == 0 || static_cast<size_t>(count) > std::numeric_limits<size_t>::max() / stride) {
            LOG_ERROR(
                "BufferPoolManager: invalid block allocation request size=%zu count=%d stride=%zu", buffer_size, count,
                stride
            );
            return 0;
        }
        size_t block_size = stride * static_cast<size_t>(count);
        void *host_base = nullptr;
        void *dev_base = alloc_and_register_block(block_size, &host_base);
        if (dev_base == nullptr) return 0;
        (void)host_base;

        size_t published = 0;
        auto dev_addr = reinterpret_cast<uintptr_t>(dev_base);
        for (int i = 0; i < count; i++) {
            void *dev_ptr = reinterpret_cast<void *>(dev_addr + stride * static_cast<size_t>(i));
            int target_shard = shard_index >= 0 ? shard_index : i;
            if (push_recycled(kind, dev_ptr, target_shard) || retire_unqueued_buffer(kind, dev_ptr, target_shard)) {
                published++;
            }
        }
        if (published == 0) {
            LOG_ERROR("BufferPoolManager: failed to publish any carved buffers from block %p", dev_base);
        }
        return published;
    }

    /**
     * Allocate a new device block and pair it with a host shadow via
     * ops_.reg. Tracks a range mapping so resolve_host_ptr() can handle
     * carved sub-buffers.
     *
     * @param size              Byte size to allocate.
     * @param[out] host_ptr_out Host shadow pointer.
     * @return                  Device pointer, or nullptr on failure.
     */
    void *alloc_and_register_block(size_t size, void **host_ptr_out) {
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
            block_ranges_.push_back(
                BlockRange{
                    reinterpret_cast<uintptr_t>(dev_ptr), reinterpret_cast<uintptr_t>(dev_ptr) + size, dev_ptr, host_ptr
                }
            );
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
        void *release_ptr = nullptr;
        if (!claim_release_pointer(dev_ptr, &release_ptr)) return;

        void *host_ptr = nullptr;
        bool free_host_shadow = false;
        {
            std::scoped_lock<std::mutex> lock(mapping_mutex_);
            auto it = dev_to_host_.find(release_ptr);
            host_ptr = (it != dev_to_host_.end()) ? it->second : nullptr;
            if (it != dev_to_host_.end()) {
                dev_to_host_.erase(it);
            }
            block_ranges_.erase(
                std::remove_if(
                    block_ranges_.begin(), block_ranges_.end(),
                    [release_ptr](const BlockRange &range) {
                        return range.dev_base == release_ptr;
                    }
                ),
                block_ranges_.end()
            );
            free_host_shadow = (host_ptr != nullptr && malloc_shadows_.erase(host_ptr) > 0);
        }
        if (ops_.free_) {
            ops_.free_(release_ptr);
        }
        {
            std::scoped_lock<std::mutex> lock(mapping_mutex_);
            released_allocations_.erase(release_ptr);
        }
        if (free_host_shadow) {
            std::free(host_ptr);
        }
    }

    /**
     * Resolve a device pointer to the host-mapped pointer recorded at
     * alloc_and_register_block / register_mapping time. Mappings are built
     * during init/proactive refill and are immutable while mgmt/collector
     * threads run, so this hot path is read-only and lock-free.
     */
    void *resolve_host_ptr(void *dev_ptr) const {
        const auto &exact_mappings = dev_to_host_;
        auto it = exact_mappings.find(dev_ptr);
        if (it != exact_mappings.end()) return it->second;
        if (void *host_ptr = resolve_host_ptr_from_range(dev_ptr); host_ptr != nullptr) {
            return host_ptr;
        }
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
     * `release_all_owned()`, and `free_buffer()` — HAL-managed mappings
     * (e.g. `halHostRegister` results) must NOT be added here.
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
        auto &pool = recycled_[shard][kind];
        void *p = nullptr;
        return pool.pop(p) ? p : nullptr;
    }

    void *pop_recycled_for_startup(int kind) {
        for (size_t shard = 0; shard < recycled_.size(); shard++) {
            if (void *p = pop_recycled(kind, static_cast<int>(shard)); p != nullptr) return p;
        }
        return nullptr;
    }

    bool push_recycled(int kind, void *dev_ptr, int shard_index = 0) {
        auto shard = normalize_shard(shard_index);
        if (!recycled_[shard][kind].push(dev_ptr)) {
            LOG_ERROR(
                "BufferPoolManager: recycled queue full for shard=%zu kind=%d capacity=%zu", shard, kind,
                kRecycledQueueCapacity
            );
            return false;
        }
        return true;
    }

    bool retire_unqueued_buffer(int kind, void *dev_ptr, int shard_index = 0) {
        auto shard = normalize_shard(shard_index);
        if (!retired_[shard][kind].push(dev_ptr)) {
            LOG_ERROR(
                "BufferPoolManager: failed to park retired buffer for shard=%zu kind=%d dev_ptr=%p", shard, kind,
                dev_ptr
            );
            return false;
        }
        return true;
    }

    size_t recycled_count(int kind, int shard_index) const {
        auto shard = normalize_shard(shard_index);
        return recycled_[shard][kind].size();
    }

    size_t recycled_count(int kind) const {
        size_t total = 0;
        for (size_t shard = 0; shard < recycled_.size(); shard++) {
            total += recycled_[shard][kind].size();
        }
        return total;
    }

    bool recycled_empty() const {
        for (size_t shard = 0; shard < recycled_.size(); shard++) {
            for (int kind = 0; kind < Module::kBufferKinds; kind++) {
                if (!recycled_[shard][kind].empty()) return false;
            }
        }
        return true;
    }

    /**
     * Drain everything currently in a done queue shard back into that shard's
     * per-kind recycled pool. Runtime callers keep this single-consumer: the
     * replenish thread is the only runtime consumer for done shards.
     */
    size_t drain_done_into_recycled(int shard_index) {
        auto &shard = done_shards_[normalize_shard(shard_index)];
        size_t drained = 0;
        DoneInfo info{};
        while (shard.queue.pop(info)) {
            if (push_recycled(info.kind, info.dev_ptr, shard_index) ||
                retire_unqueued_buffer(info.kind, info.dev_ptr, shard_index)) {
                drained++;
            }
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

    /**
     * Return the device pointer that owns this allocation. For carved buffers
     * this is the registered block base; for legacy one-buffer mappings it is
     * the pointer itself.
     */
    void *release_pointer_for(void *dev_ptr) const {
        std::scoped_lock<std::mutex> lock(mapping_mutex_);
        return allocation_base_for_locked(dev_ptr);
    }

    /**
     * Claim an allocation for release. Multiple carved sub-buffers can point
     * into the same registered block; only the first claim returns true.
     */
    bool claim_release_pointer(void *dev_ptr, void **release_ptr_out) {
        if (release_ptr_out == nullptr) return false;
        std::scoped_lock<std::mutex> lock(mapping_mutex_);
        void *release_ptr = tracked_allocation_base_for_locked(dev_ptr);
        if (release_ptr == nullptr) {
            *release_ptr_out = nullptr;
            return false;
        }
        if (!released_allocations_.insert(release_ptr).second) {
            *release_ptr_out = release_ptr;
            return false;
        }
        *release_ptr_out = release_ptr;
        return true;
    }

    void *shared_mem_dev() const { return shared_mem_dev_; }
    void *shared_mem_host() const { return shared_mem_host_; }
    int device_id() const { return device_id_; }

private:
    using ReadyRing = SpscRing<ReadyBufferInfo, kReadyQueueCapacity>;
    using DoneRing = SpscRing<DoneInfo, kPoolQueueCapacity>;
    using RecycledRing = SpscRing<void *, kRecycledQueueCapacity>;

    struct BlockRange {
        uintptr_t dev_begin;
        uintptr_t dev_end;
        void *dev_base;
        void *host_base;
    };

    struct ReadyQueueShard {
        ReadyRing queue;
        std::mutex wait_mutex;
        std::condition_variable cv;
        std::atomic<uint64_t> notify_epoch{0};
    };

    struct DoneQueueShard {
        DoneRing queue;
    };

    static size_t normalize_shard(int shard_index) {
        if (shard_index < 0) return 0;
        return static_cast<size_t>(shard_index) % static_cast<size_t>(kCollectorShardCount);
    }

    static size_t align_up(size_t value, size_t alignment) {
        if (alignment == 0) return value;
        size_t rem = value % alignment;
        if (rem == 0) return value;
        if (value > std::numeric_limits<size_t>::max() - (alignment - rem)) return 0;
        return value + (alignment - rem);
    }

    void *allocation_base_for_locked(void *dev_ptr) const {
        if (dev_ptr == nullptr) return nullptr;
        uintptr_t addr = reinterpret_cast<uintptr_t>(dev_ptr);
        for (const auto &range : block_ranges_) {
            if (addr >= range.dev_begin && addr < range.dev_end) {
                return range.dev_base;
            }
        }
        return dev_ptr;
    }

    void *tracked_allocation_base_for_locked(void *dev_ptr) const {
        if (dev_ptr == nullptr) return nullptr;
        uintptr_t addr = reinterpret_cast<uintptr_t>(dev_ptr);
        for (const auto &range : block_ranges_) {
            if (addr >= range.dev_begin && addr < range.dev_end) {
                return range.dev_base;
            }
        }
        return dev_to_host_.find(dev_ptr) != dev_to_host_.end() ? dev_ptr : nullptr;
    }

    void *resolve_host_ptr_from_range(void *dev_ptr) const {
        if (dev_ptr == nullptr) return nullptr;
        uintptr_t addr = reinterpret_cast<uintptr_t>(dev_ptr);
        for (const auto &range : block_ranges_) {
            if (addr >= range.dev_begin && addr < range.dev_end) {
                uintptr_t offset = addr - range.dev_begin;
                return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(range.host_base) + offset);
            }
        }
        return nullptr;
    }

    // Subsystem inputs (set by ProfilerBase::start via set_memory_context).
    void *shared_mem_dev_{nullptr};
    void *shared_mem_host_{nullptr};
    size_t shm_size_{0};
    int device_id_{-1};
    MemoryOps ops_;

    // mgmt → collector
    std::array<ReadyQueueShard, kCollectorShardCount> ready_shards_;

    // collector → mgmt
    std::array<DoneQueueShard, kCollectorShardCount> done_shards_;

    // Host-side pointer mappings are shared across all collector shards.
    mutable std::mutex mapping_mutex_;

    // dev → host exact mappings plus block ranges for carved buffers.
    std::unordered_map<void *, void *> dev_to_host_;
    std::vector<BlockRange> block_ranges_;
    std::unordered_set<void *> released_allocations_;

    // Host shadows the framework itself malloc'd (via
    // `default_host_shadow_register` or `ProfilerBase::alloc_paired_buffer`'s
    // copy-to-device branch). Only these are `std::free`d on teardown —
    // HAL-managed mappings (halHostRegister) live outside this set.
    std::unordered_set<void *> malloc_shadows_;

    // Local recycled buffer pools indexed by collector shard, then Module-defined kind id.
    std::array<std::array<RecycledRing, Module::kBufferKinds>, kCollectorShardCount> recycled_;

    // Error-path holding pools for buffers removed from recycled_ or popped
    // from device ready queues but not published to a collector/free_queue.
    std::array<std::array<RetiredPool, Module::kBufferKinds>, kCollectorShardCount> retired_;
};

}  // namespace profiling_common

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_BUFFER_POOL_MANAGER_H_
