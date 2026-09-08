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
 * @brief CRTP scaffolding shared by ChipSwimlane, PMU, DepGen, ArgsDump,
 *        and ScopeStats collectors.
 *
 * Owns the BufferPoolManager<Module>, drain/replenish mgmt thread(s) that
 * poll AICPU ready queues and recycle collector-done buffers, and the
 * collector poll thread(s).
 *
 * Module concept contract
 * -----------------------
 *
 * Each profiling subsystem provides a `Module` struct (e.g., ChipSwimlaneModule,
 * DumpModule, PmuModule) that supplies the data-layout traits the unified
 * mgmt-loop algorithms (ProfilerAlgorithms<Module>) need. Required members:
 *
 *   // Types
 *   using DataHeader      = ...;   // Shared-memory header (e.g. ChipSwimlaneDataHeader).
 *   using ReadyEntry      = ...;   // Per-AICPU-thread ready-queue entry.
 *   using ReadyBufferInfo = ...;   // Hand-off struct to collector thread(s)
 *                                  // (carries dev/host ptrs, optional kind
 *                                  // discriminator, and the seq).
 *   using FreeQueue       = ...;   // Per-instance SPSC queue of free buffer
 *                                  // pointers; must expose `head`, `tail`,
 *                                  // `buffer_ptrs[kSlotCount]`.
 *
 *   // Constants
 *   static constexpr int      kBufferKinds;    // ChipSwimlane=4, Dump=1, PMU=1.
 *   static constexpr uint32_t kReadyQueueSize; // Per-thread ready-queue depth.
 *   // Optional: host-side done ring depth (defaults to 1024).
 *   static constexpr uint32_t kHostPoolQueueSize;
 *   // Optional: shard-local per-kind recycled ring depth.
 *   // Defaults to kHostPoolQueueSize.
 *   static constexpr uint32_t kHostRecycledQueueSize;
 *   static constexpr uint32_t kSlotCount;      // FreeQueue::buffer_ptrs[] length.
 *   static constexpr const char* kSubsystemName; // "PMU" / "ChipSwimlane" / "Dump".
 *   // Optional: CAPACITY of the drain / collector shard arrays (defaults to
 *   // 1). Bounds the shard arrays at compile time; the number of threads
 *   // actually started is the runtime min(aicpu_thread_num,
 *   // kMaxCollectorThreads), latched by ProfilerBase::set_aicpu_thread_num().
 *   // Subsystems whose only device-side producer is the orchestrator (DepGen,
 *   // ScopeStats) set this to 1: one drain thread scans every AICPU ready
 *   // queue and finds the single producer's, so extra shards would only ever
 *   // be empty.
 *   static constexpr int      kMaxCollectorThreads;
 *   // Optional: refresh cached queue metadata before a replenish pass.
 *   template <typename Mgr>
 *   static void refresh_replenish_metadata(Mgr&, DataHeader*);
 *
 *   // Header pointer cast (host_ptr → DataHeader*)
 *   static DataHeader* header_from_shm(void* shared_mem_host);
 *
 *   // Per-kind alloc batch size for proactive_replenish's free-queue fallback.
 *   static int batch_size(int kind);
 *
 *   // Optional: steady-state low-water mark for a shard-local recycled lane.
 *   // Two forms; the two-arg one wins if both are present:
 *   //   static int recycled_warm_target(int kind);
 *   //   static int recycled_warm_target(int kind, int shard_count);
 *   // Take the two-arg form when the target depends on how many shards share
 *   // the device's cores — with `shard_count` live shards each one owns
 *   // ceil(cores / shard_count) cores, so the target must grow as shards
 *   // shrink. Omitting both means no watermark (0).
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
 *   // proactive_replenish to top up. Every instance of one kind has the same
 *   // buffer size, so completed buffers can move between collector shards.
 *   // Callback signature:
 *   //   (int kind, FreeQueue* fq, size_t buffer_size).
 *   template <typename Cb>
 *   static void for_each_instance(void* shm_host, DataHeader*, Cb&&);
 *
 * Alloc policy
 * ------------
 *
 *   process_entry          replenishes the originating free_queue from the
 *                          current drain shard's local recycled pool. It does
 *                          not allocate on the runtime hot path. When that pool
 *                          is dry it reports the site back, and the drain loop
 *                          retries it after every sweep — the only way a lane
 *                          with no buffer left to publish can recover, since
 *                          this top-up is otherwise entry-driven.
 *   proactive_replenish    fills to kSlotCount across all instances before
 *                          drain/collector threads start. If recycled is dry,
 *                          it allocates one registered block and carves it
 *                          into a batch of buffers.
 *   mgmt_replenish_loop    routes collector-done buffers to same-kind lanes
 *                          below their optional recycled watermarks, then tops
 *                          up remaining deficits in batches of
 *                          max(kSlotCount, gap). It never writes device
 *                          free_queues, so the drain hot path remains
 *                          allocation-free and owns all runtime free_queue
 *                          publication — which makes every free_queue
 *                          single-writer by structure, not by timing.
 *
 * The above two algorithms live in ProfilerAlgorithms<Module>; Module only
 * supplies the data-access traits above. Implementors must NOT zero `count`
 * (or any other AICPU-owned field) on the host side — AICPU is the sole
 * writer to those fields and resets them itself on flush/drop/pop.
 *
 * Lifecycle (the only correct teardown order):
 *   1. Derived::init() — on the success path, calls set_memory_context() to
 *      stash the alloc/reg/free callbacks, shm_dev/host pointers,
 *      shm_size and device_id on the base. If init aborts before that,
 *      start(tf) becomes a no-op (shm_host_ stays nullptr).
 *   2. start(tf) — atomically: (a) assembles a MemoryOps from the stashed
 *      callbacks, (b) hands it to the manager via set_memory_context,
 *      (c) launches the mgmt thread(s), (d) launches the collector thread(s).
 *      Mgmt is started before collectors because mgmt is the only writer to
 *      the host ready queue shard(s) and collectors are their consumers.
 *   3. ... device execution ...
 *   4. stop() — atomically:
 *        a) flips mgmt_running_, joins the mgmt thread(s); the drain thread's
 *           final-drain pass pushes the last device-ring entries into the host
 *           ready queue shard(s) before exiting.
 *        b) execution_complete_ is set; each collector loop sees it on its
 *           next idle tick, drains its host ready queue shard, and exits.
 *        c) collector thread(s) joined.
 *      Caller is then guaranteed the device-side ring and the host ready queue
 *      shard(s) are both empty and all collected data has been delivered to
 *      Derived::on_buffer_collected.
 *
 *      quiesce() gives that same guarantee without (a)'s and (c)'s joins, so
 *      the threads survive it — see its own comment.
 *
 * SVM vs host-shadow paths (chosen at runtime by the collector's MemoryOps)
 * -------------------------------------------------------------------------
 *
 *   - Collectors on platforms without SVM (a5: no halHostRegister) install
 *     `copy_to_device` / `copy_from_device` in MemoryOps so every device
 *     read/write goes through rtMemcpy (onboard) or memcpy (sim). The
 *     mgmt_loop then pulls the device-side shared-memory region into the
 *     host shadow at the top of every tick (`mirror_shm_from_device`) and
 *     pushes the few host-modified fields (`queue_heads[q]` after pop,
 *     `free_queue.tail` + `buffer_ptrs[]` after refill) back as narrow
 *     `write_range_to_device` writes. A bulk host→device write-back is
 *     intentionally avoided: it would race with AICPU writes to
 *     device-only fields (current_buf_ptr, current_buf_seq,
 *     total/dropped/mismatch counters, queue_tails, free_queue.head, and
 *     the subsystem's device-written header fields) and roll them back to the
 *     host-shadow values mirrored in at the top of the tick. Buffer
 *     contents are mirrored on demand inside ProfilerAlgorithms.
 *   - On these platforms `reg` always allocates a paired host shadow; the
 *     framework never falls back to identity-mapping (which would be wrong
 *     without SVM). Collectors pass nullptr-safe callbacks via
 *     Derived::init.
 *   - SVM platforms (a2a3: halHostRegister maps device pointers into host
 *     address space) leave `copy_to_device` / `copy_from_device` null and
 *     pass the same pointer as both shm_dev and shm_host. The mirror /
 *     write_range / copy_buffer / read_range methods then short-circuit
 *     via the manager's internal null-check and cost a single function
 *     call per tick (zero memcpy work).
 *
 * Required Derived contract
 * -------------------------
 *
 *   void on_buffer_collected(const ReadyBufferInfo& info);
 *   // Optional shard-aware overload:
 *   void on_buffer_collected(const ReadyBufferInfo& info, int collector_shard);
 *       Copy records out of `info.host_buffer_ptr` and update any
 *       per-collector state. The base class then calls
 *       `manager_.notify_copy_done(...)` so the buffer is recycled —
 *       Derived must NOT do that itself.
 *
 *   static constexpr int          kIdleTimeoutSec;
 *       Bound on how long the loop sits with no buffers AND no
 *       `execution_complete_` signal before logging an error. The collector
 *       remains alive until execution completes so a blocked drain producer
 *       always has a consumer (use the subsystem's PLATFORM_*_TIMEOUT_SECONDS).
 *
 *   static constexpr const char*  kSubsystemName;
 *       Used in the idle-timeout log line (e.g. "ChipSwimlane", "PMU", "ArgsDump").
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_HOST_PROFILER_BASE_H_
#define SRC_COMMON_PLATFORM_INCLUDE_HOST_PROFILER_BASE_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/buffer_pool_manager.h"
#include "host/profiling_copy.h"
#include "../../../worker/runtime_c_api.h"

namespace profiling_common {

template <typename Derived, typename ReadyBufferInfo, typename = void>
struct ProfilerDerivedShardAwareCollector {
    static constexpr bool value = false;
};

template <typename Derived, typename ReadyBufferInfo>
struct ProfilerDerivedShardAwareCollector<
    Derived, ReadyBufferInfo,
    std::void_t<decltype(std::declval<Derived *>()
                             ->on_buffer_collected(std::declval<const ReadyBufferInfo &>(), std::declval<int>()))>> {
    static constexpr bool value = true;
};

// Common subsystem callback signatures. All four collectors (PMU / ArgsDump
// / ChipSwimlane / DepGen) used to declare their own typedefs with identical
// shapes; these are the canonical types stashed in ProfilerBase via
// set_memory_context().
//
// Alloc / free use std::function so callers can bind state (e.g. their
// MemoryAllocator) directly via lambda capture. Register / unregister stay
// as plain function pointers — they wrap stateless HAL globals (halHost*),
// so the captureless C-callback shape matches their actual nature.
using ProfAllocCallback = std::function<void *(size_t size)>;
using ProfRegisterCallback = int (*)(void *dev_ptr, size_t size, int device_id, void **host_ptr_out);
using ProfUnregisterCallback = int (*)(void *dev_ptr, int device_id);
using ProfFreeCallback = std::function<int(void *dev_ptr)>;

// `default_host_shadow_register` was previously a free function; it has been
// folded into a lambda inside `ProfilerBase::start()` so the shadow it
// malloc's can be registered with the manager's `malloc_shadows_` set for
// safe teardown via `clear_mappings()` / `release_all_owned()`. See
// `ProfilerBase::start()` for the inline definition.

/**
 * RAII scope guard for collector `init()` rollback. On destruction (without
 * `commit()`) it (1) calls `manager.release_all_owned(release_fn)` to free
 * every framework-tracked device allocation and malloc shadow, and (2)
 * releases any extra direct dev_ptrs the collector added via `add_direct_ptr()`
 * (used for pointers the collector owns outside the framework — e.g. PMU
 * per-core `PmuAicoreRing` allocations on a5).
 *
 * Pattern:
 *   int Collector::init(...) {
 *       ...
 *       set_memory_context(...);
 *       InitRollbackGuard<Manager> guard(manager_, free_cb);
 *       void *dev_ptr = alloc_paired_buffer(size, &host_ptr);
 *       if (dev_ptr == nullptr) return PTO_RUNTIME_ERR_INTERNAL;       // guard runs, frees nothing yet
 *       ...
 *       void *direct = alloc_cb(...);
 *       guard.add_direct_ptr(direct);            // ensure it's freed on abort
 *       ...
 *       guard.commit();                          // success — disarm
 *       initialized_ = true;
 *       return 0;
 *   }
 */
template <typename Manager>
class InitRollbackGuard {
public:
    using ReleaseFn = std::function<int(void *)>;

    InitRollbackGuard(Manager &manager, ReleaseFn release_fn) :
        manager_(manager),
        release_fn_(std::move(release_fn)),
        committed_(false) {}

    ~InitRollbackGuard() {
        if (committed_) return;
        for (void *p : direct_ptrs_) {
            if (p != nullptr && release_fn_) {
                release_fn_(p);
            }
        }
        // Call release_all_owned unconditionally: it also frees malloc'd
        // host shadows (via std::free, no callback needed). Gating on
        // release_fn_ here would leak shadows if a collector ever passed
        // an empty free_cb. Device-pointer release is gated inside the
        // lambda instead.
        manager_.release_all_owned([this](void *p) {
            if (p != nullptr && release_fn_) {
                release_fn_(p);
            }
        });
    }

    InitRollbackGuard(const InitRollbackGuard &) = delete;
    InitRollbackGuard &operator=(const InitRollbackGuard &) = delete;

    void add_direct_ptr(void *p) {
        if (p != nullptr) direct_ptrs_.push_back(p);
    }
    void commit() { committed_ = true; }

private:
    Manager &manager_;
    ReleaseFn release_fn_;
    std::vector<void *> direct_ptrs_;
    bool committed_;
};

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

// Outcome of one free_queue top-up. `filled` is false when the recycled lane
// ran dry before the queue reached capacity, i.e. the site needs revisiting.
struct TopUpResult {
    uint64_t pushed;
    bool filled;
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
    //
    // a5: the head advance is written back to device immediately via
    // `mgr.write_range_to_device(&header->queue_heads[q], ...)` so AICPU sees
    // the consumer-side update without us bulk-mirroring the whole shm region
    // (which would clobber AICPU-owned fields elsewhere in the shm).
    //
    // Torn-read defense: the per-tick `mirror_shm_from_device` is a single
    // bulk rtMemcpy that is not atomic w.r.t. concurrent AICPU writes. AICPU
    // publishes a ready entry by first writing `queues[q][tail].{buffer_ptr,
    // core_index, buffer_seq}` and then bumping `queue_tails[q]`. If the
    // bulk mirror happens to scan the entry slot first and the tail counter
    // last, host can observe `head < tail` while the entry it's about to
    // read is still pre-publish (e.g. `buffer_ptr == 0`). We refresh the
    // entry with `read_range_from_device` and skip the pop if the refreshed
    // entry still looks empty — try again next tick.
    template <typename Mgr>
    static bool
    try_pop_aicpu_entry(Mgr &mgr, DataHeader *header, int q, ReadyEntry &out, bool refresh_indices = false) {
        if (refresh_indices) {
            if (mgr.read_range_from_device(&header->queue_heads[q], sizeof(header->queue_heads[q])) != 0 ||
                mgr.read_range_from_device(&header->queue_tails[q], sizeof(header->queue_tails[q])) != 0) {
                LOG_ERROR("%s: failed to refresh ready_queue indices for thread %d", Module::kSubsystemName, q);
                return false;
            }
            rmb();
        }
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

        // Re-pull this single entry from device to defeat the torn-read
        // race described above. If the entry's `buffer_ptr` is still 0 the
        // producer hasn't finished publishing — treat the queue as empty
        // for this tick.
        if (mgr.read_range_from_device(&header->queues[q][head], sizeof(header->queues[q][head])) != 0) {
            LOG_ERROR("%s: failed to refresh ready_queue entry for thread %d", Module::kSubsystemName, q);
            return false;
        }
        rmb();
        out = header->queues[q][head];
        if (out.buffer_ptr == 0) {
            return false;
        }
        uint32_t old_head = head;
        uint32_t next_head = (head + 1) % Module::kReadyQueueSize;
        header->queue_heads[q] = next_head;
        wmb();
        // Push the new head value back to device. The bulk mirror_shm_to_device
        // is intentionally not used here — see buffer_pool_manager.h.
        if (mgr.write_range_to_device(&header->queue_heads[q], sizeof(header->queue_heads[q])) != 0) {
            header->queue_heads[q] = old_head;
            LOG_ERROR("%s: failed to advance ready_queue head for thread %d", Module::kSubsystemName, q);
            return false;
        }
        return true;
    }

    // Refill the originating pool's free_queue from this drain shard's local
    // recycled pool before handing the full buffer to the collector.
    //
    // `short_site_out`, when non-null, receives this entry's site if the top-up
    // could not fill the queue (the shard's recycled lane ran dry). The caller
    // retries those sites once per sweep — see mgmt_drain_loop. Without that a
    // starved lane could never recover, because this top-up is entry-driven and
    // a lane with no buffer has nothing left to publish.
    //
    // a5 specifics: after resolving the popped buffer's host shadow, copy
    // the buffer contents from device to host before delivery. The host
    // shadow seen by the collector then matches what the device wrote.
    template <typename Mgr>
    static void
    process_entry(Mgr &mgr, DataHeader *header, int q, const ReadyEntry &entry, EntrySite<Module> *short_site_out) {
        auto site_opt = Module::resolve_entry(mgr.shared_mem_host(), header, q, entry);
        if (!site_opt.has_value()) return;
        auto &site = *site_opt;

        site.info.host_buffer_ptr = mgr.resolve_host_ptr(site.info.dev_buffer_ptr);
        if (site.info.host_buffer_ptr == nullptr) {
            // resolve_host_ptr already logged. Drop rather than deliver null.
            return;
        }
        // a5: pull buffer contents from device into the host shadow before
        // the collector reads `count` and `records[]`.
        if (mgr.copy_buffer_from_device(site.info.host_buffer_ptr, site.info.dev_buffer_ptr, site.buffer_size) != 0) {
            LOG_ERROR(
                "%s: failed to copy ready buffer from device (kind=%d, thread=%d)", Module::kSubsystemName, site.kind, q
            );
            return;
        }

        // Drain-driven free_queue top-up. The drain shard that serves ready queue
        // q is the sole runtime writer of every free_queue that q's entries
        // resolve to, so this needs no coordination with the replenish thread —
        // that thread only writes host-side recycled lanes at runtime.
        if (!top_up_free_queue(mgr, site.kind, *site.free_queue, site.buffer_size, q).filled &&
            short_site_out != nullptr) {
            *short_site_out = site;
        }

        // The device ready entry was already acknowledged by
        // try_pop_aicpu_entry(). Keep ownership here until the collector frees
        // a host-ring slot; retiring on transient host backpressure would make
        // the buffer unreachable to Derived::on_buffer_collected().
        mgr.wait_push_to_ready(site.info, q);
    }

    // Top up every (kind, instance) free_queue to kSlotCount before worker
    // threads start. At runtime each drain shard only refills its own lane.
    template <typename Mgr>
    static uint64_t proactive_replenish(Mgr &mgr, DataHeader *header) {
        uint64_t pushed = replenish_free_queues(mgr, header);
        pushed += replenish_recycled_pools(mgr, header);
        return pushed;
    }

    // Fill every (kind, instance) free_queue from any recycled lane, allocating
    // when they are dry. Startup only: `shard_index=-1` lets obtain_buffer
    // allocate and lets pop_recycled_for_startup consume from every shard, both
    // of which are safe only before the drain threads exist. At runtime the
    // owning drain shard is the sole free_queue writer.
    template <typename Mgr>
    static uint64_t replenish_free_queues(Mgr &mgr, DataHeader *header) {
        uint64_t pushed = 0;
        refresh_replenish_metadata(mgr, header, 0);
        Module::for_each_instance(mgr.shared_mem_host(), header, [&](int kind, FreeQueue *fq, size_t buf_size) {
            pushed += top_up_free_queue(mgr, kind, *fq, buf_size, /*shard_index=*/-1).pushed;
        });
        return pushed;
    }

    // Keep shard-local recycled pools above the optional Module watermark.
    // Used both by startup proactive_replenish and by the runtime replenish
    // thread. It allocates only into host-side recycled lanes; it does not
    // touch device free_queues. A small gap still allocates a slot-sized batch
    // to amortize registration; a large gap is filled in one allocation.
    template <typename Mgr>
    static uint64_t replenish_recycled_pools(Mgr &mgr, DataHeader *header) {
        std::array<size_t, Module::kBufferKinds> buffer_sizes{};
        Module::for_each_instance(mgr.shared_mem_host(), header, [&](int kind, FreeQueue *, size_t buf_size) {
            if (kind >= 0 && kind < Module::kBufferKinds && buffer_sizes[static_cast<size_t>(kind)] == 0) {
                buffer_sizes[static_cast<size_t>(kind)] = buf_size;
            }
        });

        const int shard_count = mgr.shard_count();
        uint64_t pushed = 0;
        for (int kind = 0; kind < Module::kBufferKinds; kind++) {
            if (buffer_sizes[static_cast<size_t>(kind)] == 0) continue;
            for (int shard = 0; shard < shard_count; shard++) {
                size_t target = clamped_recycled_warm_target<Mgr>(kind, shard, shard_count);
                if (target == 0) continue;
                size_t current = mgr.recycled_count(kind, shard);
                if (current >= target) continue;
                size_t gap = target - current;
                size_t batch_count = std::max(static_cast<size_t>(Module::kSlotCount), gap);
                int batch = batch_count > static_cast<size_t>(std::numeric_limits<int>::max()) ?
                                std::numeric_limits<int>::max() :
                                static_cast<int>(batch_count);
                pushed += mgr.allocate_recycled_batch(kind, buffer_sizes[static_cast<size_t>(kind)], batch, shard);
            }
        }
        return pushed;
    }

    // Retry the free_queue top-up for sites whose recycled lane ran dry, so a
    // starved lane recovers locally. Called once per drain sweep, over only the
    // sites that actually came up short — empty in the normal case, so this path
    // costs nothing until a lane runs dry.
    //
    // Runs on the owning drain shard, which is the sole runtime writer of these
    // queues; `shard_index` is the ready queue the site was observed on and
    // BufferPoolManager folds it onto that shard's recycled lane.
    template <typename Mgr>
    static bool retry_short_site(Mgr &mgr, const EntrySite<Module> &site, int shard_index) {
        return top_up_free_queue(mgr, site.kind, *site.free_queue, site.buffer_size, shard_index).filled;
    }

private:
    static int recycled_warm_target(int kind, int shard_count) {
        return profiler_module_recycled_warm_target<Module>(kind, shard_count);
    }

    template <typename Mgr>
    static size_t clamped_recycled_warm_target(int kind, int shard, int shard_count) {
        int target = recycled_warm_target(kind, shard_count);
        if (target <= 0) return 0;
        size_t requested = static_cast<size_t>(target);
        if (requested > Mgr::kRecycledQueueCapacity) {
            LOG_WARN(
                "%s: recycled warm target too large for shard=%d kind=%d: target=%zu capacity=%zu; clamping",
                Module::kSubsystemName, shard, kind, requested, Mgr::kRecycledQueueCapacity
            );
            return Mgr::kRecycledQueueCapacity;
        }
        return requested;
    }

    template <typename Mgr, typename M = Module>
    static auto refresh_replenish_metadata(Mgr &mgr, DataHeader *header, int)
        -> decltype(M::refresh_replenish_metadata(mgr, header), void()) {
        M::refresh_replenish_metadata(mgr, header);
    }

    template <typename Mgr>
    static void refresh_replenish_metadata(Mgr &, DataHeader *, long) {}

    // Fallback used by drain-shard free_queue top-up.
    template <typename Mgr>
    static void *obtain_buffer(Mgr &mgr, int kind, size_t buf_size, int shard_index) {
        if (shard_index < 0) {
            if (void *p = mgr.pop_recycled_for_startup(kind); p != nullptr) return p;
            (void)mgr.allocate_recycled_batch(kind, buf_size, Module::batch_size(kind), shard_index);
            return mgr.pop_recycled_for_startup(kind);
        }

        void *p = mgr.pop_recycled(kind, shard_index);
        if (p != nullptr) return p;

        return nullptr;
    }

    // Append one buffer pointer to a per-instance free_queue if it has
    // capacity. The queue owner is the drain shard for that AICPU producer;
    // proactive_replenish calls this only before drain threads start.
    //
    // a5: write the new slot and the advanced tail back to device via
    // `write_range_to_device` so AICPU sees the refill without us bulk
    // mirroring (which would clobber AICPU-owned fields). The slot is
    // written before the tail so AICPU never observes a tail update without
    // the corresponding pointer.
    template <typename Mgr>
    static bool try_push_to_free_queue(Mgr &mgr, FreeQueue &fq, void *dev_ptr) {
        if (mgr.read_range_from_device(&fq.head, sizeof(fq.head)) != 0) {
            LOG_ERROR("%s: failed to refresh free_queue head", Module::kSubsystemName);
            return false;
        }
        rmb();
        uint32_t fq_head = fq.head;
        uint32_t fq_tail = fq.tail;
        if (fq_tail - fq_head >= Module::kSlotCount) {
            return false;
        }
        uint32_t slot_idx = fq_tail % Module::kSlotCount;
        uint64_t old_slot = fq.buffer_ptrs[slot_idx];
        fq.buffer_ptrs[slot_idx] = reinterpret_cast<uint64_t>(dev_ptr);
        wmb();
        if (mgr.write_range_to_device(&fq.buffer_ptrs[slot_idx], sizeof(fq.buffer_ptrs[slot_idx])) != 0) {
            fq.buffer_ptrs[slot_idx] = old_slot;
            LOG_ERROR("%s: failed to publish free_queue slot", Module::kSubsystemName);
            return false;
        }
        fq.tail = fq_tail + 1;
        wmb();
        if (mgr.write_range_to_device(&fq.tail, sizeof(fq.tail)) != 0) {
            fq.tail = fq_tail;
            fq.buffer_ptrs[slot_idx] = old_slot;
            LOG_ERROR("%s: failed to publish free_queue tail", Module::kSubsystemName);
            return false;
        }
        return true;
    }

    // Unknown means the device head could not be refreshed, so whether the queue
    // has room is not established. It is deliberately distinct from Full: a
    // caller that treated it as Full would stop retrying a lane that may still be
    // starved.
    enum class QueueSpace { Available, Full, Unknown };

    template <typename Mgr>
    static QueueSpace free_queue_space(Mgr &mgr, FreeQueue &fq) {
        if (mgr.read_range_from_device(&fq.head, sizeof(fq.head)) != 0) {
            LOG_ERROR("%s: failed to refresh free_queue head", Module::kSubsystemName);
            return QueueSpace::Unknown;
        }
        rmb();
        return fq.tail - fq.head < Module::kSlotCount ? QueueSpace::Available : QueueSpace::Full;
    }

    // Fill one (kind, instance) free_queue to kSlotCount. Startup uses any
    // recycled lane and may batch-allocate; runtime uses only the drain
    // shard's local recycled lane and returns when it is dry.
    //
    // `filled` distinguishes "the queue is at capacity" from "the recycled lane
    // ran dry first", which is what tells a drain shard it must come back to this
    // site. `pushed` alone cannot: a top-up that pushed nothing may equally mean
    // the queue was already full.
    template <typename Mgr>
    static TopUpResult top_up_free_queue(Mgr &mgr, int kind, FreeQueue &fq, size_t buf_size, int shard_index = 0) {
        uint64_t pushed = 0;

        for (;;) {
            QueueSpace space = free_queue_space(mgr, fq);
            if (space == QueueSpace::Full) return {pushed, true};
            if (space == QueueSpace::Unknown) return {pushed, false};

            void *new_dev = obtain_buffer(mgr, kind, buf_size, shard_index);
            if (new_dev == nullptr) return {pushed, false};
            if (!try_push_to_free_queue(mgr, fq, new_dev)) {
                (void)mgr.retire_unqueued_buffer(kind, new_dev, shard_index);
                LOG_ERROR("%s: failed to return recycled buffer to free_queue", Module::kSubsystemName);
                return {pushed, false};
            }
            pushed++;
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

    ProfilerBase(const ProfilerBase &) = delete;
    ProfilerBase &operator=(const ProfilerBase &) = delete;

    // Per-subsystem arena acknowledgement (CRTP hook). Default: nothing to do.
    // A subsystem whose collector owns a separate reusable region (only args_dump
    // today: the per-thread payload arena) overrides this to publish, per lane,
    // how much of that region the host has consumed — the device blocks on that
    // watermark before overwriting arena bytes. Called once per replenish tick,
    // so it must be cheap and must not wait for asynchronous work.
    void publish_arena_acks() {}

private:
    friend Derived;
    ProfilerBase() = default;
    ~ProfilerBase() = default;

public:
    /**
     * Latch the runtime AICPU thread count. Must be the FIRST thing
     * Derived::init() does after validating its thread-count argument —
     * collectors seed their recycled lanes later in init() (via
     * manager_.push_recycled), and those calls already fold their shard
     * argument modulo the manager's shard count.
     *
     * Two distinct quantities come out of this:
     *
     *   queue_count_ — how many DEVICE ready queues exist, i.e. how many AICPU
     *                  threads can produce. Always `aicpu_thread_num`.
     *   shard_count_ — how many drain/collector threads (== host shards) to
     *                  run. Capped by Module::kMaxCollectorThreads, which is 1
     *                  for the orchestrator-only subsystems (DepGen,
     *                  ScopeStats): they have a single device-side producer, so
     *                  one drain thread scanning all `queue_count_` queues is
     *                  the whole job.
     *
     * The two are equal for the subsystems whose producers are the scheduler
     * threads (ChipSwimlane, ArgsDump, PMU).
     */
    void set_aicpu_thread_num(int aicpu_thread_num) {
        queue_count_ = aicpu_thread_num;
        shard_count_ = std::min(aicpu_thread_num, Manager::kMaxCollectorShards);
        manager_.set_shard_count(shard_count_);
        thread_num_set_ = true;
    }

    /**
     * Stash the memory context produced by Derived::init(). Must be called
     * on the init() success path; if init aborts before this, start(tf) is
     * a no-op.
     *
     * `copy_to_device` / `copy_from_device` are arch-specific: SVM platforms
     * (a2a3) leave them null and pass `shm_dev == shm_host`; non-SVM
     * platforms (a5) install `profiling_copy_to_device` /
     * `profiling_copy_from_device` and pass distinct shm pointers. The
     * framework picks the right register fallback (identity vs host-shadow
     * malloc) based on whether `copy_to_device` was provided.
     *
     * `register_cb` may be nullptr — start(tf) installs the appropriate
     * default for the arch path (identity on SVM platforms, host-shadow
     * malloc + memset 0 + copy_to_device on non-SVM platforms).
     */
    void set_memory_context(
        const ProfAllocCallback &alloc_cb, ProfRegisterCallback register_cb, const ProfFreeCallback &free_cb,
        std::function<int(void *, const void *, size_t)> copy_to_device,
        std::function<int(void *, const void *, size_t)> copy_from_device, void *shm_dev, void *shm_host,
        size_t shm_size, int device_id
    ) {
        alloc_cb_ = alloc_cb;
        register_cb_ = register_cb;
        free_cb_ = free_cb;
        copy_to_device_ = std::move(copy_to_device);
        copy_from_device_ = std::move(copy_from_device);
        shm_dev_ = shm_dev;
        shm_host_ = shm_host;
        shm_size_ = shm_size;
        device_id_ = device_id;
    }

    /**
     * Drop the stashed memory context. Called by Derived::finalize() so
     * that a subsequent start(tf) on a finalized collector becomes a no-op.
     */
    void clear_memory_context() {
        alloc_cb_ = nullptr;
        register_cb_ = nullptr;
        free_cb_ = nullptr;
        copy_to_device_ = nullptr;
        copy_from_device_ = nullptr;
        shm_dev_ = nullptr;
        shm_host_ = nullptr;
        shm_size_ = 0;
        device_id_ = -1;
    }

    /**
     * Assemble a MemoryOps from the callbacks stashed by set_memory_context()
     * and launch the mgmt + collector threads. If shm_host_ is nullptr (Derived's
     * init() aborted before set_memory_context, or finalize() has cleared
     * the context) this is a no-op.
     *
     * Order matters: mgmt is started before collectors because mgmt is the only
     * writer to the host ready queue shards and collectors are the consumers. The
     * register slot defaults to identity on the SVM path (copy_to_device_
     * is null) or to a host-shadow malloc lambda on the non-SVM path
     * (copy_to_device_ installed) — so BufferPoolManager always has a
     * valid reg path. The host-shadow lambda registers each malloc'd
     * shadow with `manager_.add_malloc_shadow()` so teardown can free
     * exactly the framework-owned shadows and leave HAL mappings alone.
     */
    void start(const ThreadFactory &thread_factory) {
        if (shm_host_ == nullptr) return;
        // Idempotent, like Derived::init(): the collector is resident across
        // runs, so every run's arming reaches this and only the first should
        // spawn. Without the guard each run would append another full set of
        // threads to the same collector.
        if (!collector_threads_.empty()) return;

        if (!thread_num_set_) {
            LOG_WARN(
                "%s: set_aicpu_thread_num() never called; falling back to %d shards", Derived::kSubsystemName,
                shard_count_
            );
        }
        if (shard_count_ < 1 || shard_count_ > Manager::kMaxCollectorShards || queue_count_ < 1 ||
            queue_count_ > PLATFORM_MAX_AICPU_THREADS) {
            LOG_ERROR(
                "%s: invalid thread counts (shards=%d max=%d, queues=%d max=%d); not starting", Derived::kSubsystemName,
                shard_count_, Manager::kMaxCollectorShards, queue_count_, PLATFORM_MAX_AICPU_THREADS
            );
            return;
        }

        MemoryOps ops;
        ops.alloc = alloc_cb_;
        ops.free_ = free_cb_;
        if (register_cb_ != nullptr) {
            ops.reg = register_cb_;
        } else if (copy_to_device_) {
            // Non-SVM platform: host-shadow allocate + copy zeros to device.
            // Capture `this` so the malloc'd shadow can be registered as
            // framework-owned via the manager.
            auto copy_to_device = copy_to_device_;
            ops.reg = [this,
                       copy_to_device](void *dev_ptr, size_t size, int /*device_id*/, void **host_ptr_out) -> int {
                if (host_ptr_out == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
                void *host_ptr = std::malloc(size);
                if (host_ptr == nullptr) {
                    *host_ptr_out = nullptr;
                    return PTO_RUNTIME_ERR_INTERNAL;
                }
                std::memset(host_ptr, 0, size);
                int rc = copy_to_device(dev_ptr, host_ptr, size);
                if (rc != 0) {
                    std::free(host_ptr);
                    *host_ptr_out = nullptr;
                    return rc;
                }
                manager_.add_malloc_shadow(host_ptr);
                *host_ptr_out = host_ptr;
                return 0;
            };
        } else {
            // SVM platform: identity-map (host_ptr == dev_ptr).
            ops.reg = [](void *dev_ptr, size_t /*size*/, int /*device_id*/, void **host_ptr_out) {
                *host_ptr_out = dev_ptr;
                return 0;
            };
        }
        // copy_to_device_ / copy_from_device_ may be null (SVM path); the
        // manager's internal null-checks short-circuit mirror_/range_/buffer_
        // calls to no-ops in that case.
        ops.copy_to_device = copy_to_device_;
        ops.copy_from_device = copy_from_device_;
        manager_.set_memory_context(std::move(ops), shm_dev_, shm_host_, shm_size_, device_id_);

        execution_complete_.store(false, std::memory_order_release);
        // Reset the quiescence handshake so a restarted collector cannot see a
        // previous run's acks. Safe to do unsynchronized: the std::thread
        // constructors below are the synchronization point for the workers that
        // read these.
        drain_quiesce_epoch_.store(0, std::memory_order_relaxed);
        collect_quiesce_epoch_.store(0, std::memory_order_relaxed);
        for (int i = 0; i < Manager::kMaxCollectorShards; i++) {
            drain_acked_[i].store(0, std::memory_order_relaxed);
            collect_acked_[i].store(0, std::memory_order_relaxed);
        }
        {
            DataHeader *header = Module::header_from_shm(manager_.shared_mem_host());
            (void)ProfilerAlgorithms<Module>::proactive_replenish(manager_, header);
        }

        // Drain and collector counts are the same value, so every ready shard
        // has exactly one drain-thread producer — the SPSC invariant the host
        // queues rely on.
        const int n = shard_count_;

        mgmt_running_.store(true, std::memory_order_release);
        mgmt_drain_threads_.reserve(n);
        for (int i = 0; i < n; i++) {
            if (thread_factory) {
                mgmt_drain_threads_.push_back(thread_factory([this, i, n]() {
                    mgmt_drain_loop(i, n);
                }));
            } else {
                mgmt_drain_threads_.emplace_back(&ProfilerBase::mgmt_drain_loop, this, i, n);
            }
        }
        if (thread_factory) {
            mgmt_replenish_thread_ = thread_factory([this]() {
                mgmt_replenish_loop();
            });
        } else {
            mgmt_replenish_thread_ = std::thread(&ProfilerBase::mgmt_replenish_loop, this);
        }

        collector_threads_.reserve(n);
        for (int i = 0; i < n; i++) {
            if (thread_factory) {
                collector_threads_.push_back(thread_factory([this, i]() {
                    poll_and_collect_loop(i);
                }));
            } else {
                collector_threads_.emplace_back(&ProfilerBase::poll_and_collect_loop, this, i);
            }
        }
    }

    /**
     * Drain to a quiescent point without retiring the threads. On return the
     * device-side ring and the host ready queue shard(s) are empty and
     * Derived::on_buffer_collected has been called for every entry that was in
     * either — the same guarantee stop() gives, minus the thread teardown.
     *
     * Precondition: the device-side producers have stopped. A drain worker
     * reports its shard quiescent after one full sweep that found nothing, so a
     * producer still writing could push a record in behind that report.
     * Callers already satisfy this — the run is drained before teardown.
     *
     * Idempotent, and a no-op before start() or after stop(). Like stop(), it
     * waits without a deadline: a wedged worker hangs the caller here exactly
     * as it would hang the join in stop().
     */
    void quiesce() {
        if (collector_threads_.empty()) return;
        const int n = shard_count_;

        // Phase one: mgmt sweeps the device-side ring into the host shards.
        const uint64_t epoch = drain_quiesce_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
        wait_for_epoch(drain_acked_, n, epoch);

        // Phase two: only now can a collector's "my shard is empty" mean the
        // pipeline is empty rather than that mgmt has not pushed yet.
        collect_quiesce_epoch_.store(epoch, std::memory_order_release);
        wait_for_epoch(collect_acked_, n, epoch);
    }

    /**
     * Stop the drain/replenish mgmt threads, drain whatever the drain side
     * pushes during its final pass, and join the collector. Idempotent. Caller
     * is guaranteed on return that mgmt's device-side ringbuffer and the
     * host-side ready queue shard(s) are empty and Derived::on_buffer_collected
     * has been called for every entry that was in either queue. Framework-owned
     * buffers are NOT freed here — Derived's finalize() must do that.
     *
     * Order matters: stop+join mgmt first so its final-drain pass is fully
     * landed in the host shards BEFORE we tell poll to exit. Otherwise mgmt's
     * last batch has no consumer.
     */
    void stop() {
        mgmt_running_.store(false, std::memory_order_release);
        for (auto &thread : mgmt_drain_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        mgmt_drain_threads_.clear();
        if (mgmt_replenish_thread_.joinable()) {
            mgmt_replenish_thread_.join();
        }
        execution_complete_.store(true, std::memory_order_release);
        for (auto &thread : collector_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        collector_threads_.clear();
    }

    Manager &manager() { return manager_; }
    const Manager &manager() const { return manager_; }

protected:
    Manager manager_;
    std::atomic<bool> execution_complete_{false};
    std::vector<std::thread> collector_threads_;

    // Latched by set_aicpu_thread_num() during Derived::init(); read by the
    // threads start() spawns. The std::thread constructor is the
    // synchronization point, so plain ints need no atomic.
    int queue_count_{PLATFORM_MAX_AICPU_THREADS};
    int shard_count_{Manager::kMaxCollectorShards};
    bool thread_num_set_{false};

    // Memory context stashed by Derived::init() via set_memory_context().
    // Derived may read these from finalize() / alloc helpers via the
    // inherited names. ProfilerBase owns the lifetime: Derived must call
    // clear_memory_context() in finalize() to drop them.
    ProfAllocCallback alloc_cb_{nullptr};
    ProfRegisterCallback register_cb_{nullptr};
    ProfFreeCallback free_cb_{nullptr};
    // copy_to_device_ / copy_from_device_ are set by non-SVM platforms
    // (a5) to profiling_copy_* wrappers; left null by SVM platforms (a2a3)
    // so the manager's mirror methods short-circuit to no-ops.
    std::function<int(void *, const void *, size_t)> copy_to_device_;
    std::function<int(void *, const void *, size_t)> copy_from_device_;
    void *shm_dev_{nullptr};
    void *shm_host_{nullptr};
    size_t shm_size_{0};
    int device_id_{-1};

    /**
     * RAII counterpart of ``alloc_single_buffer``: unregister the host
     * mapping (if there is one) then release the device memory. Each
     * Derived's ``finalize()`` funnels every release site through here so
     * the framework never frees a dev_ptr without first taking down the
     * matching ``halHostRegister`` slot. On a5 onboard ``register_cb`` is
     * always nullptr so the unregister branch is a no-op — the helper is
     * shared with a2a3 anyway for code uniformity.
     */
    void release_one_buffer(void *dev_ptr, ProfUnregisterCallback unregister_cb, const ProfFreeCallback &free_cb) {
        if (dev_ptr == nullptr) return;
        void *release_ptr = nullptr;
        if (!manager_.claim_release_pointer(dev_ptr, &release_ptr)) return;
        if (unregister_cb != nullptr) {
            int rc = unregister_cb(release_ptr, device_id_);
            if (rc != 0) {
                LOG_ERROR("halHostUnregister failed for dev_ptr %p: %d", release_ptr, rc);
            }
        }
        if (free_cb) {
            free_cb(release_ptr);
        }
    }

    /**
     * Allocate a device buffer and its paired host view, picking the right
     * pairing strategy based on the memory context stashed by
     * set_memory_context():
     *
     *   - `register_cb_` set       (a2a3 onboard): `register_cb_(dev, …)`
     *     installs the halHostRegister mapping; host_ptr is the
     *     identity-mapped view of the same memory.
     *   - non-SVM platform (a5):   `copy_to_device_` is installed →
     *     malloc a paired host shadow, zero it, push the zeros to the
     *     device side. The host shadow lives until BufferPoolManager teardown
     *     via `clear_mappings()` or `release_all_owned()`.
     *   - SVM platform (a2a3 sim): `register_cb_` null AND `copy_to_device_`
     *     null → identity-map (host_ptr == dev_ptr).
     *
     * On any failure the device pointer is freed via `free_cb_` and
     * nullptr is returned; on success the dev↔host mapping is registered
     * with the buffer pool so resolve_host_ptr() finds it later.
     *
     * Used by leaf collectors' init() to allocate the shared-memory header
     * region and any per-instance buffers, replacing the per-arch ad-hoc
     * branch trees they used to carry.
     */
    void *alloc_paired_buffer(size_t size, void **host_ptr_out) {
        if (host_ptr_out == nullptr) return nullptr;
        *host_ptr_out = nullptr;
        if (!alloc_cb_) return nullptr;

        void *dev_ptr = alloc_cb_(size);
        if (dev_ptr == nullptr) return nullptr;

        void *host_ptr = nullptr;
        if (register_cb_ != nullptr) {
            int rc = register_cb_(dev_ptr, size, device_id_, &host_ptr);
            if (rc != 0 || host_ptr == nullptr) {
                LOG_ERROR("ProfilerBase::alloc_paired_buffer: register_cb_ failed: %d", rc);
                if (free_cb_) free_cb_(dev_ptr);
                return nullptr;
            }
        } else if (copy_to_device_) {
            // Non-SVM: malloc + zero + push to device.
            host_ptr = std::malloc(size);
            if (host_ptr == nullptr) {
                LOG_ERROR("ProfilerBase::alloc_paired_buffer: host shadow alloc failed for %zu bytes", size);
                if (free_cb_) free_cb_(dev_ptr);
                return nullptr;
            }
            std::memset(host_ptr, 0, size);
            int rc = copy_to_device_(dev_ptr, host_ptr, size);
            if (rc != 0) {
                LOG_ERROR("ProfilerBase::alloc_paired_buffer: copy_to_device failed: %d", rc);
                std::free(host_ptr);
                if (free_cb_) free_cb_(dev_ptr);
                return nullptr;
            }
            manager_.add_malloc_shadow(host_ptr);
        } else {
            // SVM: identity-map.
            host_ptr = dev_ptr;
        }

        *host_ptr_out = host_ptr;
        manager_.register_mapping(dev_ptr, host_ptr);
        return dev_ptr;
    }

private:
    // Teardown-path wait, so a sleep is permitted here: no task's latency
    // passes through it (codestyle.md rule 5 exempts teardown).
    template <typename Acks>
    static void wait_for_epoch(const Acks &acks, int n, uint64_t epoch) {
        for (int i = 0; i < n; i++) {
            while (acks[i].load(std::memory_order_acquire) != epoch) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
    }

    void mgmt_drain_loop(int queue_start, int queue_stride) {
        DataHeader *header = Module::header_from_shm(manager_.shared_mem_host());
        using Alg = ProfilerAlgorithms<Module>;
        constexpr int kIdleBusyPollLoops = 64;
        int idle_busy_polls = 0;

        // Sites whose last top-up ran out of recycled buffers, keyed by the ready
        // queue they were seen on. Thread-local to this shard: every site here
        // resolved from an entry on one of this shard's queues, and each queue is
        // served by exactly one shard, so this shard is their sole writer.
        std::vector<std::pair<int, EntrySite<Module>>> short_sites;

        while (mgmt_running_.load(std::memory_order_relaxed)) {
            bool found_any = false;
            for (int q = queue_start; q < queue_count_; q += queue_stride) {
                ReadyEntry entry;
                while (Alg::try_pop_aicpu_entry(manager_, header, q, entry, true)) {
                    // A null free_queue is the "nothing to retry" sentinel;
                    // process_entry only writes this on a short top-up.
                    EntrySite<Module> short_site{};
                    Alg::process_entry(manager_, header, q, entry, &short_site);
                    if (short_site.free_queue != nullptr) {
                        record_short_site(short_sites, q, short_site);
                    }
                    found_any = true;
                }
            }
            if (found_any) {
                idle_busy_polls = 0;
            }

            // Retry after every sweep, not only on an idle one: a lane that has
            // run dry has nothing left to publish, so it would otherwise wait
            // behind a busy sibling on this same shard indefinitely.
            retry_short_sites(short_sites);

            // A full sweep that found nothing means this worker's slice of the
            // device-side queues is empty. With producers stopped (quiesce()'s
            // precondition) nothing can arrive behind this report, so it is the
            // quiescent condition for phase one.
            if (!found_any) {
                const uint64_t requested = drain_quiesce_epoch_.load(std::memory_order_acquire);
                if (drain_acked_[queue_start].load(std::memory_order_relaxed) != requested) {
                    drain_acked_[queue_start].store(requested, std::memory_order_release);
                }
            }

            if (!found_any) {
                if (idle_busy_polls < kIdleBusyPollLoops) {
                    idle_busy_polls++;
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            }
        }

        for (int q = queue_start; q < queue_count_; q += queue_stride) {
            ReadyEntry entry;
            while (Alg::try_pop_aicpu_entry(manager_, header, q, entry, true)) {
                Alg::process_entry(manager_, header, q, entry, nullptr);
            }
        }
    }

    // Append a site unless this shard is already tracking that free_queue. The
    // list is bounded by the shard's instance count, so a linear scan is cheaper
    // than any keyed container at these sizes.
    static void record_short_site(
        std::vector<std::pair<int, EntrySite<Module>>> &short_sites, int q, const EntrySite<Module> &site
    ) {
        for (const auto &tracked : short_sites) {
            if (tracked.second.free_queue == site.free_queue) return;
        }
        short_sites.emplace_back(q, site);
    }

    void retry_short_sites(std::vector<std::pair<int, EntrySite<Module>>> &short_sites) {
        using Alg = ProfilerAlgorithms<Module>;
        for (size_t i = 0; i < short_sites.size();) {
            if (Alg::retry_short_site(manager_, short_sites[i].second, short_sites[i].first)) {
                short_sites[i] = short_sites.back();
                short_sites.pop_back();
            } else {
                i++;
            }
        }
    }

    void mgmt_replenish_loop() {
        DataHeader *header = Module::header_from_shm(manager_.shared_mem_host());
        using Alg = ProfilerAlgorithms<Module>;
        while (mgmt_running_.load(std::memory_order_relaxed)) {
            size_t drained = manager_.drain_done_into_recycled();
            uint64_t replenished = Alg::replenish_recycled_pools(manager_, header);

            // This thread's only runtime writes are to host-side recycled lanes:
            // the owning drain shard publishes into device free_queues. Keeping
            // that split is what makes each free_queue single-writer by structure.
            //
            // The arena acknowledgement (CRTP): a no-op for every subsystem but
            // args_dump, which publishes its per-lane payload watermark here.
            static_cast<Derived *>(this)->publish_arena_acks();

            if (drained == 0 && replenished == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    }

    /**
     * Main collector loop. Blocks on one manager ready-queue shard with a 100 ms
     * cv-wait tick. On each hit it dispatches the buffer to Derived via
     * on_buffer_collected() and recycles the buffer. Exits only after:
     *
     *   execution_complete_ was set (by stop()) and this ready_queue shard is
     *   empty, after a final non-blocking drain pass.
     *
     * No buffer for `Derived::kIdleTimeoutSec` after traffic is still reported
     * as a hang warning, but the consumer stays alive. A later final-drain push
     * may be blocked on this shard, so exiting before execution_complete_ would
     * leave the management thread waiting forever.
     */
    void poll_and_collect_loop(int shard_index) {
        const auto wait_tick = std::chrono::milliseconds(100);
        const auto idle_timeout = std::chrono::seconds(Derived::kIdleTimeoutSec);
        std::optional<std::chrono::steady_clock::time_point> idle_start;
        bool has_seen_buffer = false;

        while (true) {
            ReadyBufferInfo info;
            if (manager_.wait_pop_ready(info, wait_tick, shard_index)) {
                consume(info, shard_index);
                has_seen_buffer = true;
                idle_start.reset();
                continue;
            }
            if (execution_complete_.load(std::memory_order_acquire)) {
                while (manager_.try_pop_ready(info, shard_index)) {
                    consume(info, shard_index);
                    has_seen_buffer = true;
                }
                break;
            }
            // Phase two of the quiescence handshake. wait_pop_ready timed out,
            // so this shard is empty; mgmt has already reported its own sweep
            // done for this epoch, so nothing further can arrive. Placed above
            // the has_seen_buffer guard below: a shard that never received a
            // buffer is a valid run shape and still has to report, or quiesce()
            // would wait on it forever.
            {
                const uint64_t requested = collect_quiesce_epoch_.load(std::memory_order_acquire);
                if (collect_acked_[shard_index].load(std::memory_order_relaxed) != requested) {
                    while (manager_.try_pop_ready(info, shard_index)) {
                        consume(info, shard_index);
                        has_seen_buffer = true;
                    }
                    collect_acked_[shard_index].store(requested, std::memory_order_release);
                }
            }
            // A shard that has never seen a buffer is a valid run shape at any
            // shard count — a subsystem can legitimately emit nothing for a
            // whole run. execution_complete_ above is the exit path for that
            // case; the idle timeout below only guards a shard that saw traffic
            // and then stalled.
            if (!has_seen_buffer) {
                continue;
            }
            if (!idle_start.has_value()) {
                idle_start = std::chrono::steady_clock::now();
            }
            if (std::chrono::steady_clock::now() - idle_start.value() >= idle_timeout) {
                LOG_ERROR(
                    "%s collector idle timeout after %d seconds — staying alive until execution completes",
                    Derived::kSubsystemName, Derived::kIdleTimeoutSec
                );
                // Report once per traffic burst. A newly consumed buffer sets
                // has_seen_buffer again and re-arms the detector.
                has_seen_buffer = false;
                idle_start.reset();
            }
        }
    }

    void consume(const ReadyBufferInfo &info, int shard_index) {
        if constexpr (ProfilerDerivedShardAwareCollector<Derived, ReadyBufferInfo>::value) {
            static_cast<Derived *>(this)->on_buffer_collected(info, shard_index);
        } else {
            static_cast<Derived *>(this)->on_buffer_collected(info);
        }
        if constexpr (Module::kBufferKinds > 1) {
            (void)manager_.notify_copy_done(info.dev_buffer_ptr, Module::kind_of(info), shard_index);
        } else {
            (void)manager_.notify_copy_done(info.dev_buffer_ptr, 0, shard_index);
        }
    }

    std::vector<std::thread> mgmt_drain_threads_;
    std::thread mgmt_replenish_thread_;
    std::atomic<bool> mgmt_running_{false};

    // Two-phase quiescence handshake. Each phase is a monotonic epoch the
    // caller publishes and every worker of that phase echoes back once it has
    // reached the quiescent condition for its own shard.
    //
    // The phases are ordered, not concurrent: a collector that reported its
    // shard empty before mgmt finished its sweep would be reporting on a queue
    // mgmt is still about to push into. So collect_quiesce_epoch_ is published
    // only after every drain ack for that epoch has landed.
    std::atomic<uint64_t> drain_quiesce_epoch_{0};
    std::atomic<uint64_t> collect_quiesce_epoch_{0};
    std::array<std::atomic<uint64_t>, Manager::kMaxCollectorShards> drain_acked_{};
    std::array<std::atomic<uint64_t>, Manager::kMaxCollectorShards> collect_acked_{};
};

}  // namespace profiling_common

#endif  // SRC_COMMON_PLATFORM_INCLUDE_HOST_PROFILER_BASE_H_
