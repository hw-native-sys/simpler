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
 * @file profiler_base.h
 * @brief CRTP scaffolding shared by L2Perf / Dump / PMU collectors.
 *
 * Owns the BufferPoolManager<Module>, the mgmt thread (which polls AICPU
 * ready queues and recycles buffers), and the collector poll thread.
 *
 * Module concept contract
 * -----------------------
 *
 * Each profiling subsystem provides a `Module` struct (e.g., L2PerfModule,
 * DumpModule, PmuModule) that supplies the data-layout traits the unified
 * mgmt-loop algorithms (ProfilerAlgorithms<Module>) need. Required members:
 *
 *   // Types
 *   using DataHeader      = ...;   // Shared-memory header (e.g. L2PerfDataHeader).
 *   using ReadyEntry      = ...;   // Per-AICPU-thread ready-queue entry.
 *   using ReadyBufferInfo = ...;   // Hand-off struct to the collector thread
 *                                  // (carries dev/host ptrs, optional kind
 *                                  // discriminator, and the seq).
 *   using FreeQueue       = ...;   // Per-instance SPSC queue of free buffer
 *                                  // pointers; must expose `head`, `tail`,
 *                                  // `buffer_ptrs[kSlotCount]`.
 *
 *   // Constants
 *   static constexpr int      kBufferKinds;    // L2Perf=2 (perf+phase), Dump=1, PMU=1.
 *   static constexpr uint32_t kReadyQueueSize; // Per-thread ready-queue depth.
 *   static constexpr uint32_t kSlotCount;      // FreeQueue::buffer_ptrs[] length.
 *   static constexpr const char* kSubsystemName; // "PMU" / "L2Perf" / "Dump".
 *
 *   // Header pointer cast (host_ptr → DataHeader*)
 *   static DataHeader* header_from_shm(void* shared_mem_host);
 *
 *   // Per-kind alloc batch size for proactive_replenish's batch-alloc fallback.
 *   static int batch_size(int kind);
 *
 *   // Required only when kBufferKinds > 1: discriminate which recycled bin
 *   // a finished buffer belongs to. Single-kind modules omit this method;
 *   // ProfilerBase::consume passes 0 unconditionally for them.
 *   static int kind_of(const ReadyBufferInfo& info);
 *
 *   // Resolve a popped ReadyEntry into the originating BufferState's
 *   // free_queue + the partially-filled ReadyBufferInfo. Algorithm fills in
 *   // host_buffer_ptr after a resolve_host_ptr lookup. Return std::nullopt
 *   // to drop the entry (e.g. invalid index).
 *   static std::optional<EntrySite<Module>> resolve_entry(
 *       void* shm_host, DataHeader*, int q, const ReadyEntry&);
 *
 *   // Enumerate every (kind, instance) free_queue and its buffer size for
 *   // proactive_replenish to top up. Callback signature:
 *   //   (int kind, FreeQueue* fq, size_t buffer_size).
 *   template <typename Cb>
 *   static void for_each_instance(void* shm_host, DataHeader*, Cb&&);
 *
 * Alloc policy
 * ------------
 *
 *   process_entry          replenishes the originating free_queue with EXACTLY
 *                          one buffer per call, matching the 1-in / 1-out
 *                          ratio against the entry the AICPU just produced.
 *                          Single allocation when both recycled and done are
 *                          dry; bounds the per-tick latency.
 *   proactive_replenish    fills to kSlotCount across all instances of every
 *                          kind. When recycled drains it batch-allocates
 *                          `batch_size(kind)` buffers at once to amortize the
 *                          allocator cost — recovery from a double-empty
 *                          condition takes one tick instead of N.
 *
 * The above two algorithms live in ProfilerAlgorithms<Module>; Module only
 * supplies the data-access traits above. Implementors must NOT zero `count`
 * (or any other AICPU-owned field) on the host side — AICPU is the sole
 * writer to those fields and resets them itself on flush/drop/pop.
 *
 * Lifecycle (the only correct teardown order):
 *   1. Derived::init() — on the success path, calls set_memory_context() to
 *      stash the alloc/reg/free callbacks, user_data, shm_host pointer and
 *      device_id on the base. If init aborts before that, start(tf) becomes
 *      a no-op (shm_host_ stays nullptr).
 *   2. start(tf) — atomically: (a) assembles a MemoryOps from the stashed
 *      callbacks, (b) hands it to the manager via set_memory_context,
 *      (c) launches the mgmt thread, (d) launches the poll thread. Mgmt is
 *      started before poll because mgmt is the only writer to L2 (the
 *      ready_queue) and poll is its sole consumer.
 *   3. ... device execution ...
 *   4. stop() — atomically:
 *        a) flips mgmt_running_, joins the mgmt thread; the mgmt thread's
 *           final-drain pass pushes the last L1→L2 entries before exiting.
 *        b) execution_complete_ is set; the poll loop sees it on its next
 *           idle tick, drains L2 (which now contains mgmt's final-drain
 *           output), and exits.
 *        c) collector thread joined.
 *      Caller is then guaranteed L1 and L2 are both empty and all collected
 *      data has been delivered to Derived::on_buffer_collected.
 *
 * Required Derived contract
 * -------------------------
 *
 *   void on_buffer_collected(const ReadyBufferInfo& info);
 *       Copy records out of `info.host_buffer_ptr` and update any per-collector
 *       state. The base class then calls `manager_.notify_copy_done(...)` so
 *       the buffer is recycled — Derived must NOT do that itself.
 *
 *   static constexpr int          kIdleTimeoutSec;
 *       Bound on how long the loop sits with no buffers AND no
 *       `execution_complete_` signal before logging an error and exiting
 *       (use the subsystem's PLATFORM_*_TIMEOUT_SECONDS).
 *
 *   static constexpr const char*  kSubsystemName;
 *       Used in the idle-timeout log line (e.g. "L2Perf", "PMU", "TensorDump").
 */

#ifndef SRC_A2A3_PLATFORM_INCLUDE_HOST_PROFILING_COMMON_PROFILER_BASE_H_
#define SRC_A2A3_PLATFORM_INCLUDE_HOST_PROFILING_COMMON_PROFILER_BASE_H_

#include <atomic>
#include <chrono>
#include <functional>
#include <optional>
#include <thread>
#include <utility>

#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/profiling_common/buffer_pool_manager.h"

namespace profiling_common {

// Common subsystem callback signatures. All three collectors (PMU / TensorDump
// / L2Perf) used to declare their own typedefs with identical shapes; these
// are the canonical types stashed in ProfilerBase via set_memory_context().
using ProfAllocCallback = void *(*)(size_t size, void *user_data);
using ProfRegisterCallback = int (*)(void *dev_ptr, size_t size, int device_id, void **host_ptr_out);
using ProfFreeCallback = int (*)(void *dev_ptr, void *user_data);

// Result of Module::resolve_entry. Carries everything the unified
// process_entry algorithm needs to (a) refill the originating pool's free
// queue and (b) hand the ready buffer off to the collector.
//
//   kind        — recycled-pool index in [0, Module::kBufferKinds).
//   free_queue  — the originating pool's SPSC queue to refill with one buffer.
//   buffer_size — bytes to allocate if the recycled+done fallbacks are dry.
//   info        — partially-filled ReadyBufferInfo (dev_buffer_ptr, buffer_seq,
//                 and any module-specific index fields are set; algorithm fills
//                 host_buffer_ptr after a resolve_host_ptr lookup).
template <typename Module>
struct EntrySite {
    int kind;
    typename Module::FreeQueue *free_queue;
    size_t buffer_size;
    typename Module::ReadyBufferInfo info;
};

// Unified mgmt-loop algorithms parameterized on Module's data-access traits.
// Module supplies the layout (constants + types + resolve_entry +
// for_each_instance); ProfilerAlgorithms supplies the control flow that used
// to be hand-rolled per subsystem.
template <typename Module>
struct ProfilerAlgorithms {
    using DataHeader = typename Module::DataHeader;
    using ReadyEntry = typename Module::ReadyEntry;
    using ReadyBufferInfo = typename Module::ReadyBufferInfo;
    using FreeQueue = typename Module::FreeQueue;

    // Pop one entry from the per-thread ready queue, advancing the head with
    // the appropriate memory barriers. Returns false if the queue is empty
    // (or the device wrote an out-of-range head/tail, which is treated as
    // empty and reported).
    static bool try_pop_aicpu_entry(DataHeader *header, int q, ReadyEntry &out) {
        uint32_t head = header->queue_heads[q];
        uint32_t tail = header->queue_tails[q];
        if (head >= Module::kReadyQueueSize || tail >= Module::kReadyQueueSize) {
            LOG_ERROR(
                "%s: invalid queue indices for thread %d: head=%u tail=%u (max=%u)", Module::kSubsystemName, q, head,
                tail, Module::kReadyQueueSize
            );
            return false;
        }
        if (head == tail) return false;
        // Order the tail-vs-empty check before the entry read so the
        // entry load cannot be speculated past it on aarch64.
        rmb();

        out = header->queues[q][head];
        head = (head + 1) % Module::kReadyQueueSize;
        header->queue_heads[q] = head;
        wmb();
        return true;
    }

    // Refill the originating pool's free_queue with exactly one buffer
    // (recycled → drain done → alloc), then push the popped buffer's
    // ReadyBufferInfo to the collector LAST. Skips the push if host_ptr
    // resolution fails — handing a null pointer to on_buffer_collected would
    // crash the collector thread.
    template <typename Mgr>
    static void process_entry(Mgr &mgr, DataHeader *header, int q, const ReadyEntry &entry) {
        auto site_opt = Module::resolve_entry(mgr.shared_mem_host(), header, q, entry);
        if (!site_opt.has_value()) return;
        auto &site = *site_opt;

        void *new_dev = obtain_buffer(mgr, site.kind, site.buffer_size);
        if (new_dev != nullptr) {
            push_to_free_queue(*site.free_queue, new_dev);
        }

        site.info.host_buffer_ptr = mgr.resolve_host_ptr(site.info.dev_buffer_ptr);
        if (site.info.host_buffer_ptr == nullptr) {
            // resolve_host_ptr already logged. Drop rather than deliver null.
            return;
        }
        mgr.push_to_ready(site.info);
    }

    // Drain done_queue into recycled, then top up every (kind, instance)
    // free_queue to kSlotCount. When the recycled pool of a given kind drains
    // mid-fill, batch-allocate `batch_size(kind)` buffers and continue.
    template <typename Mgr>
    static void proactive_replenish(Mgr &mgr, DataHeader *header) {
        mgr.drain_done_into_recycled();
        Module::for_each_instance(mgr.shared_mem_host(), header, [&](int kind, FreeQueue *fq, size_t buf_size) {
            top_up_free_queue(mgr, kind, *fq, buf_size);
        });
    }

private:
    // Three-level fallback used by process_entry's 1-in/1-out replenish.
    template <typename Mgr>
    static void *obtain_buffer(Mgr &mgr, int kind, size_t buf_size) {
        void *p = mgr.pop_recycled(kind);
        if (p != nullptr) return p;
        mgr.drain_done_into_recycled();
        p = mgr.pop_recycled(kind);
        if (p != nullptr) return p;

        void *host_ptr = nullptr;
        p = mgr.alloc_and_register(buf_size, &host_ptr);
        if (p == nullptr) {
            LOG_WARN(
                "%s: alloc failed for %zu bytes (kind=%d) — increase BUFFERS_PER_* to reduce drops",
                Module::kSubsystemName, buf_size, kind
            );
        }
        return p;
    }

    // Append one buffer pointer to a per-instance free_queue. Caller owns
    // the "queue is not full" guarantee (process_entry: 1-in/1-out;
    // top_up_free_queue: explicit fq_used < kSlotCount).
    static void push_to_free_queue(FreeQueue &fq, void *dev_ptr) {
        uint32_t fq_tail = fq.tail;
        fq.buffer_ptrs[fq_tail % Module::kSlotCount] = reinterpret_cast<uint64_t>(dev_ptr);
        wmb();
        fq.tail = fq_tail + 1;
    }

    // Fill one (kind, instance) free_queue to kSlotCount, batch-allocating
    // when the recycled pool of this kind drains mid-fill.
    template <typename Mgr>
    static void top_up_free_queue(Mgr &mgr, int kind, FreeQueue &fq, size_t buf_size) {
        rmb();
        uint32_t fq_head = fq.head;
        uint32_t fq_tail = fq.tail;
        uint32_t fq_used = fq_tail - fq_head;

        while (fq_used < Module::kSlotCount) {
            void *new_dev = mgr.pop_recycled(kind);
            if (new_dev == nullptr) {
                const int batch = Module::batch_size(kind);
                for (int i = 0; i < batch; i++) {
                    void *host_ptr = nullptr;
                    void *dev = mgr.alloc_and_register(buf_size, &host_ptr);
                    if (dev == nullptr) break;
                    mgr.push_recycled(kind, dev);
                }
                new_dev = mgr.pop_recycled(kind);
            }
            if (new_dev == nullptr) return;

            fq.buffer_ptrs[fq_tail % Module::kSlotCount] = reinterpret_cast<uint64_t>(new_dev);
            wmb();
            fq_tail++;
            fq.tail = fq_tail;
            wmb();
            fq_used++;
        }
    }
};

template <typename Derived, typename Module>
class ProfilerBase {
public:
    using Manager = BufferPoolManager<Module>;
    using DataHeader = typename Module::DataHeader;
    using ReadyEntry = typename Module::ReadyEntry;
    using ReadyBufferInfo = typename Module::ReadyBufferInfo;

    ProfilerBase() = default;
    ~ProfilerBase() = default;

    ProfilerBase(const ProfilerBase &) = delete;
    ProfilerBase &operator=(const ProfilerBase &) = delete;

    /**
     * Stash the memory context produced by Derived::init(). Must be called on
     * the init() success path; if init aborts before this, start(tf) is a
     * no-op.
     */
    void set_memory_context(
        ProfAllocCallback alloc_cb, ProfRegisterCallback register_cb, ProfFreeCallback free_cb, void *user_data,
        void *shm_host, int device_id
    ) {
        alloc_cb_ = alloc_cb;
        register_cb_ = register_cb;
        free_cb_ = free_cb;
        user_data_ = user_data;
        shm_host_ = shm_host;
        device_id_ = device_id;
    }

    /**
     * Drop the stashed memory context. Called by Derived::finalize() so that
     * a subsequent start(tf) on a finalized collector becomes a no-op.
     */
    void clear_memory_context() {
        alloc_cb_ = nullptr;
        register_cb_ = nullptr;
        free_cb_ = nullptr;
        user_data_ = nullptr;
        shm_host_ = nullptr;
        device_id_ = -1;
    }

    /**
     * Assemble a MemoryOps from the callbacks stashed by set_memory_context()
     * and launch the mgmt + poll threads. If shm_host_ is nullptr (Derived's
     * init() aborted before set_memory_context, or finalize() has cleared the
     * context) this is a no-op.
     *
     * Order matters: mgmt is started before poll because mgmt is the only
     * writer to L2 and poll is its sole consumer. If register_cb_ is nullptr
     * (sim mode), an identity wrapper is installed so BufferPoolManager has a
     * single uniform code path: dev_ptr is reflected back as host_ptr.
     */
    void start(const ThreadFactory &thread_factory) {
        if (shm_host_ == nullptr) return;

        MemoryOps ops;
        ops.alloc = [cb = alloc_cb_, ud = user_data_](size_t size) {
            return cb(size, ud);
        };
        ops.free_ = [cb = free_cb_, ud = user_data_](void *dev_ptr) {
            return cb(dev_ptr, ud);
        };
        if (register_cb_ != nullptr) {
            ops.reg = register_cb_;
        } else {
            ops.reg = [](void *dev_ptr, size_t /*size*/, int /*device_id*/, void **host_ptr_out) {
                *host_ptr_out = dev_ptr;
                return 0;
            };
        }
        manager_.set_memory_context(std::move(ops), shm_host_, device_id_);

        mgmt_running_.store(true, std::memory_order_release);
        mgmt_thread_ = thread_factory([this]() {
            mgmt_loop();
        });

        execution_complete_.store(false, std::memory_order_release);
        collector_thread_ = thread_factory([this]() {
            poll_and_collect_loop();
        });
    }

    /**
     * Stop the mgmt thread, drain whatever it pushes during its final pass,
     * and join the collector. Idempotent. Caller is guaranteed on return that
     * mgmt's L1 ringbuffer and the host-side L2 ready_queue are both empty
     * and Derived::on_buffer_collected has been called for every entry that
     * was in either queue. Framework-owned buffers are NOT freed here —
     * Derived's finalize() must do that.
     *
     * Order matters: stop+join mgmt first so its final-drain pass is fully
     * landed in L2 BEFORE we tell poll to exit. Otherwise mgmt's last batch
     * has no consumer.
     */
    void stop() {
        mgmt_running_.store(false, std::memory_order_release);
        if (mgmt_thread_.joinable()) {
            mgmt_thread_.join();
        }
        execution_complete_.store(true, std::memory_order_release);
        if (collector_thread_.joinable()) {
            collector_thread_.join();
        }
    }

    Manager &manager() { return manager_; }
    const Manager &manager() const { return manager_; }

protected:
    Manager manager_;
    std::atomic<bool> execution_complete_{false};
    std::thread collector_thread_;

    // Memory context stashed by Derived::init() via set_memory_context().
    // Derived may read these from finalize() / alloc helpers via the inherited
    // names. ProfilerBase owns the lifetime: Derived must call
    // clear_memory_context() in finalize() to drop them.
    ProfAllocCallback alloc_cb_{nullptr};
    ProfRegisterCallback register_cb_{nullptr};
    ProfFreeCallback free_cb_{nullptr};
    void *user_data_{nullptr};
    void *shm_host_{nullptr};
    int device_id_{-1};

private:
    /**
     * mgmt thread main loop. Each tick:
     *   1) Drains done_queue into recycled pools.
     *   2) Iterates AICPU per-thread ready queues
     *      (PLATFORM_MAX_AICPU_THREADS upper bound; empty queues are O(1)
     *      head==tail checks) and calls Module::process_entry per entry.
     *   3) Calls Module::proactive_replenish to top up any depleted free
     *      queues.
     *   4) Sleeps 10 us if no work was done.
     *
     * On exit (mgmt_running_ → false) it does one final drain pass without
     * sleeping to flush any straggler entries the device pushed before
     * stopping.
     */
    void mgmt_loop() {
        DataHeader *header = Module::header_from_shm(manager_.shared_mem_host());
        using Alg = ProfilerAlgorithms<Module>;

        while (mgmt_running_.load(std::memory_order_acquire)) {
            manager_.drain_done_into_recycled();

            bool found_any = false;
            for (int q = 0; q < PLATFORM_MAX_AICPU_THREADS; q++) {
                ReadyEntry entry;
                while (Alg::try_pop_aicpu_entry(header, q, entry)) {
                    Alg::process_entry(manager_, header, q, entry);
                    found_any = true;
                }
            }

            Alg::proactive_replenish(manager_, header);

            if (!found_any) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }

        // Final drain after mgmt_running_ flipped: don't sleep, don't replenish.
        for (int q = 0; q < PLATFORM_MAX_AICPU_THREADS; q++) {
            ReadyEntry entry;
            while (Alg::try_pop_aicpu_entry(header, q, entry)) {
                Alg::process_entry(manager_, header, q, entry);
            }
        }
    }

    /**
     * Main collector loop. Blocks on the manager's ready_queue with a 100 ms
     * cv-wait tick. On each hit it dispatches the buffer to Derived via
     * on_buffer_collected() and recycles the buffer. Exits in two cases:
     *
     *   1. execution_complete_ was set (by stop()) and the ready_queue is
     *      empty, after a final non-blocking drain pass.
     *   2. No buffer arrived for `Derived::kIdleTimeoutSec` consecutive seconds
     *      AND execution_complete_ has not been signalled — this is a hang
     *      detector that logs an error and bails out.
     */
    void poll_and_collect_loop() {
        const auto wait_tick = std::chrono::milliseconds(100);
        const auto idle_timeout = std::chrono::seconds(Derived::kIdleTimeoutSec);
        std::optional<std::chrono::steady_clock::time_point> idle_start;

        while (true) {
            ReadyBufferInfo info;
            if (manager_.wait_pop_ready(info, wait_tick)) {
                consume(info);
                idle_start.reset();
                continue;
            }
            if (execution_complete_.load(std::memory_order_acquire)) {
                while (manager_.try_pop_ready(info)) {
                    consume(info);
                }
                break;
            }
            if (!idle_start.has_value()) {
                idle_start = std::chrono::steady_clock::now();
            }
            if (std::chrono::steady_clock::now() - idle_start.value() >= idle_timeout) {
                LOG_ERROR(
                    "%s collector idle timeout after %d seconds — giving up", Derived::kSubsystemName,
                    Derived::kIdleTimeoutSec
                );
                break;
            }
        }
    }

    void consume(const ReadyBufferInfo &info) {
        static_cast<Derived *>(this)->on_buffer_collected(info);
        if constexpr (Module::kBufferKinds > 1) {
            manager_.notify_copy_done(info.dev_buffer_ptr, Module::kind_of(info));
        } else {
            manager_.notify_copy_done(info.dev_buffer_ptr, 0);
        }
    }

    std::thread mgmt_thread_;
    std::atomic<bool> mgmt_running_{false};
};

}  // namespace profiling_common

#endif  // SRC_A2A3_PLATFORM_INCLUDE_HOST_PROFILING_COMMON_PROFILER_BASE_H_
