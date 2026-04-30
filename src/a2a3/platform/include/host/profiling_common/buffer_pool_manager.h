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
 * alloc/reg/free callbacks), and DoneInfo (per-buffer ownership info passed
 * through done_queue).
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_HOST_PROFILING_COMMON_BUFFER_POOL_MANAGER_H_
#define SRC_A2A3_PLATFORM_INCLUDE_HOST_PROFILING_COMMON_BUFFER_POOL_MANAGER_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
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
 * - alloc:     allocate `size` bytes of device memory; return nullptr on failure.
 * - reg:       register dev_ptr for host visibility, writing the host-mapped
 *              pointer to *host_ptr_out. ProfilerBase::start installs an
 *              identity wrapper here when collectors leave it empty (sim mode),
 *              so callers may invoke it unconditionally.
 * - free_:     free a previously allocated device pointer.
 */
struct MemoryOps {
    std::function<void *(size_t)> alloc;
    std::function<int(void *dev_ptr, size_t size, int device_id, void **host_ptr_out)> reg;
    std::function<int(void *dev_ptr)> free_;
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
     * resolve_host_ptr / drain_done_into_recycled triggered by the mgmt loop) is
     * invoked. Must NOT be called concurrently with the mgmt thread.
     *
     * @param ops             Memory-op callbacks (alloc/reg/free).
     * @param shared_mem_host Host-mapped base of the subsystem's shared memory.
     * @param device_id       Forwarded to ops.reg.
     */
    void set_memory_context(MemoryOps ops, void *shared_mem_host, int device_id) {
        ops_ = std::move(ops);
        shared_mem_host_ = shared_mem_host;
        device_id_ = device_id;
    }

    /**
     * Release every device buffer the framework currently owns: recycled
     * pools, done_queue, and ready_queue. Buffers still in the per-pool
     * free_queue or held as current_buf_ptr are NOT touched — those belong
     * to the collector and must be released by it (the AICPU may still be
     * referencing them via shared memory until execution ends).
     *
     * `release_fn(dev_ptr)` is invoked once per unique pointer; the collector
     * is expected to do `unregister + free` (onboard) or just `free` (sim).
     *
     * dev_to_host_ entries for the released buffers are erased; entries for
     * still-live buffers remain so resolve_host_ptr works during the
     * collector's own free_queue/current_buf_ptr cleanup pass.
     *
     * Only safe to call after ProfilerBase::stop() has joined the mgmt thread.
     */
    template <typename ReleaseFn>
    void release_owned_buffers(const ReleaseFn &release_fn) {
        std::unordered_map<void *, bool> seen;
        auto release_once = [&](void *p) {
            if (p == nullptr) return;
            if (seen.emplace(p, true).second) {
                release_fn(p);
                dev_to_host_.erase(p);
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
     * further resolve_host_ptr() lookups expected.
     */
    void clear_mappings() { dev_to_host_.clear(); }

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
     * Allocate a new buffer and register it for host access. Tracks the
     * resulting dev→host mapping so resolve_host_ptr() can find it on
     * subsequent ready-queue pops. In sim mode, the identity reg installed
     * by ProfilerBase::start makes host_ptr == dev_ptr.
     *
     * @param size              Byte size to allocate.
     * @param[out] host_ptr_out Host-mapped pointer (== dev_ptr in sim mode).
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
            free_buffer(dev_ptr);
            *host_ptr_out = nullptr;
            return nullptr;
        }
        *host_ptr_out = host_ptr;
        dev_to_host_[dev_ptr] = host_ptr;
        return dev_ptr;
    }

    void free_buffer(void *dev_ptr) {
        if (dev_ptr != nullptr && ops_.free_) {
            dev_to_host_.erase(dev_ptr);
            ops_.free_(dev_ptr);
        }
    }

    /**
     * Resolve a device pointer to the host-mapped pointer recorded at
     * alloc_and_register / register_mapping time. In sim mode the identity
     * mapping makes the lookup return dev_ptr.
     */
    void *resolve_host_ptr(void *dev_ptr) {
        auto it = dev_to_host_.find(dev_ptr);
        if (it != dev_to_host_.end()) return it->second;
        LOG_ERROR("BufferPoolManager: no host mapping for dev_ptr=%p", dev_ptr);
        return nullptr;
    }

    /**
     * Register an externally-allocated mapping. Used by the Collector during
     * initialize() when it pre-allocates buffers and wants the mgmt thread to
     * be able to resolve them later.
     */
    void register_mapping(void *dev_ptr, void *host_ptr) { dev_to_host_[dev_ptr] = host_ptr; }

    /**
     * Pull from the recycled pool of the given kind, or return nullptr if empty.
     * Caller is responsible for resolving host_ptr (via resolve_host_ptr) and
     * resetting any buffer-specific state (e.g., count = 0) before handing the
     * buffer back to AICPU.
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
     * Drain everything currently in done_queue back into the per-kind recycled
     * pool. May be called from Module::process_entry when its primary recycled
     * pool ran out, to harvest buffers the collector freed in the meantime.
     */
    void drain_done_into_recycled() {
        std::scoped_lock<std::mutex> lock(done_mutex_);
        while (!done_queue_.empty()) {
            const DoneInfo &info = done_queue_.front();
            recycled_[info.kind].push_back(info.dev_ptr);
            done_queue_.pop();
        }
    }

    void *shared_mem_host() const { return shared_mem_host_; }
    int device_id() const { return device_id_; }

private:
    // Subsystem inputs (set by ProfilerBase::start via set_memory_context).
    void *shared_mem_host_{nullptr};
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

#endif  // SRC_A2A3_PLATFORM_INCLUDE_HOST_PROFILING_COMMON_BUFFER_POOL_MANAGER_H_
