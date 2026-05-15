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
 * @brief Generic buffer-pool data structure shared by L2Perf, TensorDump,
 *        and PMU collectors. Owns:
 *
 *   - ready_queue (mgmt → collector) with mutex/cv,
 *   - done_queue (collector → mgmt) with mutex,
 *   - per-kind recycled-buffer pools,
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
 * passed through done_queue).
 *
 * a5 vs a2a3 differences (intentional)
 * ------------------------------------
 *
 * a5 has no `halHostRegister`, so device↔host transfers go through
 * rtMemcpy (onboard) or memcpy (sim) against a paired host shadow. Two
 * mechanical changes capture that:
 *
 *   1. MemoryOps carries `copy_to_device` and `copy_from_device` in
 *      addition to the {alloc, reg, free_} of a2a3. The mgmt loop calls
 *      `mirror_shm_from_device` once per tick to refresh the host shadow,
 *      then writes back only the fields it actually modifies via
 *      `write_range_to_device(field_ptr, sizeof(field))`. The bulk
 *      `mirror_shm_to_device` is kept for init/teardown but is NOT used by
 *      the mgmt loop — bulk write-back races with AICPU writes to
 *      device-only fields (current_buf_ptr, total/dropped/mismatch
 *      counters, queue_tails, free_queue.head, AicpuPhaseHeader::magic).
 *   2. `reg` allocates a paired host shadow (instead of mapping a HAL view
 *      onto the device pointer); `release_owned_buffers` therefore frees
 *      both the device pointer (via `release_fn`) and the host shadow
 *      (`free()` on the value stashed in `dev_to_host_`).
 *
 * Buffer-content mirroring (the per-tick shm copy moves header +
 * BufferStates only — not the bulk records) is done on demand inside
 * ProfilerAlgorithms::process_entry, which calls
 * `manager_.copy_buffer_from_device(host_buf, dev_buf, buffer_size)` after
 * resolving the host pointer.
 */

#ifndef SRC_A5_PLATFORM_INCLUDE_HOST_PROFILING_COMMON_BUFFER_POOL_MANAGER_H_
#define SRC_A5_PLATFORM_INCLUDE_HOST_PROFILING_COMMON_BUFFER_POOL_MANAGER_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <unordered_map>
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
 * Per-buffer ownership info threaded through the done_queue so that the mgmt
 * thread, when it recycles a finished buffer, knows which per-kind pool it
 * came from.
 */
struct DoneInfo {
    void *dev_ptr;
    int kind;  // [0, Module::kBufferKinds)
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

    BufferPoolManager() :
        recycled_(Module::kBufferKinds) {}
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
     * pools, done_queue, and ready_queue. Buffers still in the per-pool
     * free_queue or held as current_buf_ptr are NOT touched — those belong
     * to the collector and must be released by it (the AICPU may still be
     * referencing them via shared memory until execution ends).
     *
     * For each unique device pointer freed, the paired host shadow recorded
     * in `dev_to_host_` is also `free()`d before the mapping is erased. On
     * a5 every shadow comes from `malloc()` (either via ops_.reg or via
     * collector-side `alloc_single_buffer`), so the unconditional free is
     * correct.
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
                // a5: free the paired host shadow (malloc'd by ops_.reg or by
                // alloc_single_buffer). Skip the self-free corner case where
                // dev_ptr and host_ptr happen to alias.
                if (host_ptr != nullptr && host_ptr != p) {
                    std::free(host_ptr);
                }
                if (it != dev_to_host_.end()) {
                    dev_to_host_.erase(it);
                }
            }
        };

        for (auto &pool : recycled_) {
            for (void *p : pool)
                release_once(p);
            pool.clear();
        }
        {
            std::scoped_lock<std::mutex> lock(done_mutex_);
            while (!done_queue_.empty()) {
                release_once(done_queue_.front().dev_ptr);
                done_queue_.pop();
            }
        }
        {
            std::scoped_lock<std::mutex> lock(ready_mutex_);
            while (!ready_queue_.empty()) {
                release_once(ready_queue_.front().dev_buffer_ptr);
                ready_queue_.pop();
            }
        }
    }

    /**
     * Drop the dev↔host mapping table — call after the collector has freed
     * its share of buffers (free_queue + current_buf_ptr) and there are no
     * further resolve_host_ptr() lookups expected. Frees host shadows that
     * are still mapped (collectors may have invoked free_cb on the dev
     * pointer without going through release_owned_buffers).
     */
    void clear_mappings() {
        for (auto &kv : dev_to_host_) {
            if (kv.second != nullptr && kv.second != kv.first) {
                std::free(kv.second);
            }
        }
        dev_to_host_.clear();
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
     * Push the host-side modifications (advanced `queue_heads`, refilled
     * free_queues) back to the device. Called at the bottom of every mgmt
     * tick.
     *
     * NOTE: deprecated for a5 — bulk write_back races with AICPU writes to
     * device-owned fields (BufferState::current_buf_ptr, total/dropped/mismatch
     * counters, queue_tails, free_queue.head, AicpuPhaseHeader::magic, ...).
     * The bulk write rolls those updates back to whatever was in the host
     * shadow at mirror_from_device time. Keep the method around so callers
     * outside the mgmt loop (init/teardown) still have a way to push the
     * whole region, but the mgmt loop now uses `write_field_to_device` /
     * `write_range_to_device` for the few fields host actually modifies.
     */
    int mirror_shm_to_device() {
        if (shared_mem_host_ == nullptr || shared_mem_dev_ == nullptr || shm_size_ == 0) {
            return 0;
        }
        if (!ops_.copy_to_device) return 0;
        return ops_.copy_to_device(shared_mem_dev_, shared_mem_host_, shm_size_);
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
        const auto *host_field = static_cast<const char *>(const_cast<const void *>(host_field_ptr));
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
        const auto *host_field = static_cast<const char *>(const_cast<const void *>(host_field_ptr));
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
     * Pull a single buffer's contents (e.g. an L2PerfBuffer / PmuBuffer /
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
    // ready_queue: mgmt thread pushes, collector thread pops
    // -------------------------------------------------------------------------

    void push_to_ready(const ReadyBufferInfo &info) {
        {
            std::scoped_lock<std::mutex> lock(ready_mutex_);
            ready_queue_.push(info);
        }
        ready_cv_.notify_one();
    }

    bool try_pop_ready(ReadyBufferInfo &out) {
        std::scoped_lock<std::mutex> lock(ready_mutex_);
        if (ready_queue_.empty()) return false;
        out = ready_queue_.front();
        ready_queue_.pop();
        return true;
    }

    bool wait_pop_ready(ReadyBufferInfo &out, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(ready_mutex_);
        if (!ready_cv_.wait_for(lock, timeout, [this] {
                return !ready_queue_.empty();
            })) {
            return false;
        }
        out = ready_queue_.front();
        ready_queue_.pop();
        return true;
    }

    // -------------------------------------------------------------------------
    // done_queue: collector thread reports buffers it has finished copying;
    // mgmt thread folds them back into the recycled pool of the right kind.
    // -------------------------------------------------------------------------

    void notify_copy_done(void *dev_ptr, int kind) {
        std::scoped_lock<std::mutex> lock(done_mutex_);
        done_queue_.push(DoneInfo{dev_ptr, kind});
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
        dev_to_host_[dev_ptr] = host_ptr;
        return dev_ptr;
    }

    /**
     * Free a device pointer + paired host shadow tracked in dev_to_host_.
     * Currently unused by the mgmt loop (recycle path keeps buffers alive)
     * but kept for symmetry with a2a3.
     */
    void free_buffer(void *dev_ptr) {
        if (dev_ptr == nullptr) return;
        auto it = dev_to_host_.find(dev_ptr);
        void *host_ptr = (it != dev_to_host_.end()) ? it->second : nullptr;
        if (it != dev_to_host_.end()) {
            dev_to_host_.erase(it);
        }
        if (ops_.free_) {
            ops_.free_(dev_ptr);
        }
        if (host_ptr != nullptr && host_ptr != dev_ptr) {
            std::free(host_ptr);
        }
    }

    /**
     * Resolve a device pointer to the host-mapped pointer recorded at
     * alloc_and_register / register_mapping time.
     */
    void *resolve_host_ptr(void *dev_ptr) {
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
    void register_mapping(void *dev_ptr, void *host_ptr) { dev_to_host_[dev_ptr] = host_ptr; }

    /**
     * Pull from the recycled pool of the given kind, or return nullptr if
     * empty. Caller is responsible for resolving host_ptr (via
     * resolve_host_ptr) before handing the buffer back to AICPU.
     */
    void *pop_recycled(int kind) {
        auto &pool = recycled_[kind];
        if (pool.empty()) return nullptr;
        void *p = pool.back();
        pool.pop_back();
        return p;
    }

    void push_recycled(int kind, void *dev_ptr) { recycled_[kind].push_back(dev_ptr); }

    bool recycled_empty() const {
        for (const auto &pool : recycled_) {
            if (!pool.empty()) return false;
        }
        return true;
    }

    /**
     * Drain everything currently in done_queue back into the per-kind
     * recycled pool. May be called from Module::process_entry when its
     * primary recycled pool ran out, to harvest buffers the collector freed
     * in the meantime.
     */
    void drain_done_into_recycled() {
        std::scoped_lock<std::mutex> lock(done_mutex_);
        while (!done_queue_.empty()) {
            const DoneInfo &info = done_queue_.front();
            recycled_[info.kind].push_back(info.dev_ptr);
            done_queue_.pop();
        }
    }

    void *shared_mem_dev() const { return shared_mem_dev_; }
    void *shared_mem_host() const { return shared_mem_host_; }
    int device_id() const { return device_id_; }

private:
    // Subsystem inputs (set by ProfilerBase::start via set_memory_context).
    void *shared_mem_dev_{nullptr};
    void *shared_mem_host_{nullptr};
    size_t shm_size_{0};
    int device_id_{-1};
    MemoryOps ops_;

    // mgmt → collector
    std::mutex ready_mutex_;
    std::condition_variable ready_cv_;
    std::queue<ReadyBufferInfo> ready_queue_;

    // collector → mgmt
    std::mutex done_mutex_;
    std::queue<DoneInfo> done_queue_;

    // dev → host mapping (single source of truth for resolve_host_ptr)
    std::unordered_map<void *, void *> dev_to_host_;

    // Per-kind recycled buffer pools (vector indexed by Module's BufferKind id)
    std::vector<std::vector<void *>> recycled_;
};

}  // namespace profiling_common

#endif  // SRC_A5_PLATFORM_INCLUDE_HOST_PROFILING_COMMON_BUFFER_POOL_MANAGER_H_
