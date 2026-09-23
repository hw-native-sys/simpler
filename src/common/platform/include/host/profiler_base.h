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
 *   // host_buffer_ptr after a resolve_host_ptr lookup. Return std::nullopt to
 *   // reject the entry (e.g. invalid index); the drain path then retires and
 *   // counts it rather than delivering it.
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
 * These algorithms live in ProfilerAlgorithms<Module>; Module only
 * supplies the data-access traits above. Implementors must NOT zero `count`
 * (or any other AICPU-owned field) on the host side — AICPU is the sole
 * writer to those fields and resets them itself on flush/drop/pop.
 *
 * Drain-path ownership
 * --------------------
 *
 * A device ready entry is acknowledged — `queue_heads[q]` advanced — only once
 * its payload is in the host shadow, so the device owns the slot for as long as
 * the host might still fail to read it and a transport failure costs a retry
 * rather than the record. Two outcomes end that retry loop:
 *
 *   - An entry that cannot be resolved, or whose buffer this manager never
 *     mapped, can never be read; retrying it would block the queue forever and
 *     quiesce() would never complete. It is acknowledged and counted.
 *   - An entry whose copy keeps failing is acknowledged and counted once it has
 *     held the queue head for kStalledDrainEntryTimeout.
 *
 * Either way `drain_dropped_buffers()` records it, so a reconcile gap is
 * attributable to the host instead of being an anonymous silent loss, and the
 * buffer goes back to the free_queue it came from (or to the manager's retired
 * pool when that queue is full) so the pool does not shrink. The unresolvable
 * entry is the exception: its indices did not validate, so its buffer is never
 * published into a device-visible free_queue — it is parked in the retired pool
 * when the manager can map it, and withheld entirely when it cannot.
 *
 * Lifecycle (the only correct teardown order):
 *   1. Derived::init() — on the success path, calls set_memory_context() to
 *      stash the alloc/reg/free callbacks, shm_dev/host pointers,
 *      shm_size and device_id on the base and bind the manager's memory
 *      operations to that region. If init aborts before that,
 *      start(tf) becomes a no-op (shm_host_ stays nullptr).
 *   2. start(tf) — launches the mgmt thread(s), then the collector thread(s).
 *      Mgmt is started before collectors because mgmt is the only writer to
 *      the host ready queue shard(s) and collectors are their consumers.
 *   3. ... device execution ...
 *   4. stop() — atomically:
 *        a) flips mgmt_running_, joins the mgmt thread(s); the drain thread's
 *           final-drain pass pushes the last device-ring entries into the host
 *           ready queue shard(s) before exiting.
 *        b) execution_complete_ is set and the ready-queue waiters are
 *           notified; each collector drains its host ready queue shard and
 *           exits.
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

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/memory_barrier.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host/buffer_pool_manager.h"
#include "host/chip_swimlane_runs.h"
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
// folded into `bind_manager_memory_context()` so its malloc'd shadow can
// be registered with the manager's `malloc_shadows_` set for
// safe teardown via `clear_mappings()` / `release_all_owned()`. See
// `ProfilerBase::bind_manager_memory_context()` for the inline definition.

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
                // The status is observed, not discarded: a rollback that could
                // not free what it allocated leaves memory held, and the
                // manager's paired occupancy has to keep reflecting it.
                if (int rc = release_fn_(p); rc != 0) manager_.note_release_failed(p, rc);
            }
        }
        // Call release_all_owned unconditionally: it also frees malloc'd
        // host shadows (via std::free, no callback needed). Gating on
        // release_fn_ here would leak shadows if a collector ever passed
        // an empty free_cb. Device-pointer release is gated inside the
        // lambda instead.
        manager_.release_all_owned([this](void *p) {
            if (p != nullptr && release_fn_) {
                if (int rc = release_fn_(p); rc != 0) manager_.note_release_failed(p, rc);
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

// Outcome of one drain-path ready entry. kRetry is the only one that leaves the
// device's consumer index where it was, so the same entry is seen again by the
// next peek; the other two have advanced it.
enum class EntryOutcome {
    kDelivered,  // in the host hand-off ring, on its way to the collector
    kRetry,      // neither acknowledged nor delivered
    kDropped,    // acknowledged and counted as lost; never reached a collector
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

    // Read the entry at the head of the per-thread ready queue without
    // acknowledging it. Returns false if the queue is empty (or the device wrote
    // an out-of-range head/tail, which is treated as empty and reported).
    //
    // The device still owns the slot on return: only ack_aicpu_entry() advances
    // the consumer index, and process_entry calls it once the entry's payload is
    // in the host shadow. So a host-side failure between the two costs a repeat
    // of this peek rather than the record.
    //
    // Torn-read defense: the per-tick `mirror_shm_from_device` is a single
    // bulk rtMemcpy that is not atomic w.r.t. concurrent AICPU writes. AICPU
    // publishes a ready entry by first writing `queues[q][tail].{buffer_ptr,
    // core_index, buffer_seq}` and then bumping `queue_tails[q]`. If the
    // bulk mirror happens to scan the entry slot first and the tail counter
    // last, host can observe `head < tail` while the entry it's about to
    // read is still pre-publish (e.g. `buffer_ptr == 0`). We refresh the
    // entry with `read_range_from_device` and skip the peek if the refreshed
    // entry still looks empty — try again next tick.
    template <typename Mgr>
    static bool
    try_peek_aicpu_entry(Mgr &mgr, DataHeader *header, int q, ReadyEntry &out, bool refresh_indices = false) {
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
        return out.buffer_ptr != 0;
    }

    // Hand the slot holding ready queue q's head entry back to the device by
    // advancing the consumer index.
    //
    // a5: the head advance is written back to device immediately via
    // `mgr.write_range_to_device(&header->queue_heads[q], ...)` so AICPU sees
    // the consumer-side update without us bulk-mirroring the whole shm region
    // (which would clobber AICPU-owned fields elsewhere in the shm). A failed
    // write-back leaves the host-side index where the device has it, so the
    // entry stays unacknowledged and the next peek sees it again.
    template <typename Mgr>
    static bool ack_aicpu_entry(Mgr &mgr, DataHeader *header, int q) {
        const uint32_t old_head = header->queue_heads[q];
        header->queue_heads[q] = (old_head + 1) % Module::kReadyQueueSize;
        wmb();
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
    // `retries_exhausted` says this entry has held the queue head long enough
    // that a still-failing copy must be retired rather than retried again; see
    // ProfilerBase::kStalledDrainEntryTimeout.
    //
    // a5 specifics: after resolving the popped buffer's host shadow, copy
    // the buffer contents from device to host before delivery. The host
    // shadow seen by the collector then matches what the device wrote.
    template <typename Mgr>
    static EntryOutcome process_entry(
        Mgr &mgr, DataHeader *header, int q, const ReadyEntry &entry, EntrySite<Module> *short_site_out,
        bool retries_exhausted
    ) {
        auto site_opt = Module::resolve_entry(mgr.shared_mem_host(), header, q, entry);
        if (!site_opt.has_value()) {
            // resolve_entry already logged which index failed to validate, and no
            // retry can make it validate.
            if (!ack_aicpu_entry(mgr, header, q)) return EntryOutcome::kRetry;
            // The buffer pointer travelled in the same entry, so it is trustworthy
            // only as far as the manager can vouch for it, and never trustworthy
            // enough to publish into a device-visible free_queue for AICPU to
            // dereference. A pointer inside a block this manager owns is parked in
            // the host-only retired pool, which teardown releases; one the manager
            // cannot map is withheld entirely, because release_pointer_for() hands
            // an unmapped pointer to the free callback unchanged, and freeing a
            // pointer from a corrupt entry is worse than forgetting it. The kind is
            // unknown here and 0 is a bucket label only: the retired pools are
            // drained by iterating every shard and kind, never selectively.
            void *dev_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
            const bool parked =
                mgr.resolve_host_ptr(dev_ptr) != nullptr && mgr.retire_unqueued_buffer(/*kind=*/0, dev_ptr, q);
            LOG_ERROR(
                "%s: retired an unresolvable ready entry on thread %d; its records are lost and its buffer %p is %s",
                Module::kSubsystemName, q, dev_ptr, parked ? "parked until teardown" : "withheld from the pool"
            );
            return EntryOutcome::kDropped;
        }
        auto &site = *site_opt;

        site.info.host_buffer_ptr = mgr.resolve_host_ptr(site.info.dev_buffer_ptr);
        if (site.info.host_buffer_ptr == nullptr) {
            // resolve_host_ptr already logged. Mappings are established when a
            // buffer is allocated, i.e. before the device can ever publish it,
            // so an unmappable buffer will not become mappable on a retry.
            return retire_undeliverable_entry(mgr, header, q, site, "its device buffer is not mapped on this host");
        }
        // a5: pull buffer contents from device into the host shadow before
        // the collector reads `count` and `records[]`.
        if (mgr.copy_buffer_from_device(site.info.host_buffer_ptr, site.info.dev_buffer_ptr, site.buffer_size) != 0) {
            LOG_ERROR(
                "%s: failed to copy ready buffer from device (kind=%d, thread=%d)", Module::kSubsystemName, site.kind, q
            );
            if (!retries_exhausted) return EntryOutcome::kRetry;
            return retire_undeliverable_entry(mgr, header, q, site, "its device buffer could not be copied to host");
        }

        // The payload is in the host shadow, so the device's slot can go back.
        // Doing this before the copy would put a failure between the
        // acknowledgement and the delivery, which is exactly how a record gets
        // lost without anything counting it.
        if (!ack_aicpu_entry(mgr, header, q)) return EntryOutcome::kRetry;

        // Drain-driven free_queue top-up. The drain shard that serves ready queue
        // q is the sole runtime writer of every free_queue that q's entries
        // resolve to, so this needs no coordination with the replenish thread —
        // that thread only writes host-side recycled lanes at runtime.
        if (!top_up_free_queue(mgr, site.kind, *site.free_queue, site.buffer_size, q).filled &&
            short_site_out != nullptr) {
            *short_site_out = site;
        }

        // Ownership stays here until the collector frees a host-ring slot;
        // retiring on transient host backpressure would make the buffer
        // unreachable to Derived::on_buffer_collected().
        mgr.wait_push_to_ready(site.info, q);
        return EntryOutcome::kDelivered;
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

    // Retire a ready entry whose payload the host will never read: acknowledge
    // the device slot so the queue keeps draining, then put the buffer back into
    // the free_queue it came from so the pool does not shrink. That queue's sole
    // runtime writer is this drain shard, which is why the buffer can go straight
    // back rather than through a recycled lane another thread owns; when it has
    // no room the manager's retired pool holds the buffer until teardown frees
    // it, exactly as a failed top-up does.
    template <typename Mgr>
    static EntryOutcome
    retire_undeliverable_entry(Mgr &mgr, DataHeader *header, int q, const EntrySite<Module> &site, const char *reason) {
        if (!ack_aicpu_entry(mgr, header, q)) return EntryOutcome::kRetry;
        LOG_ERROR(
            "%s: retired ready buffer %p (kind=%d, thread=%d) because %s; its records are lost and are part of this "
            "run's reconcile gap",
            Module::kSubsystemName, site.info.dev_buffer_ptr, site.kind, q, reason
        );
        if (!try_push_to_free_queue(mgr, *site.free_queue, site.info.dev_buffer_ptr)) {
            (void)mgr.retire_unqueued_buffer(site.kind, site.info.dev_buffer_ptr, q);
        }
        return EntryOutcome::kDropped;
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
     * `register_cb` may be nullptr — set_memory_context() installs the appropriate
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
        // begin_run() publishes fields before start(). Bind now so a rebuilt
        // collector cannot translate new host pointers through the previous
        // region's base. Init/rebind requires the collector threads to be stopped.
        bind_manager_memory_context();
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
        manager_.clear_memory_context();
    }

    /**
     * Launch the mgmt + collector threads. If shm_host_ is nullptr (Derived's
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
        manager_.notify_ready_waiters();
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
        manager_.notify_ready_waiters();
        for (auto &thread : collector_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        collector_threads_.clear();
    }

    Manager &manager() { return manager_; }
    const Manager &manager() const { return manager_; }

    /**
     * Push one host-shadow field to its device mirror, naming the subsystem and
     * the field if the write is rejected.
     *
     * `write_range_to_device` rejects a field outside the manager's shm window
     * and returns non-zero. A caller that discards that result configures
     * nothing and reports nothing: the device keeps its previous value, and the
     * only trace is the manager's own log line, which names neither the
     * subsystem nor what was being published. That is how a base-pointer
     * disagreement between a collector's `shm_host_` and the manager's copy
     * reaches a reader as "no records were produced" (#2206).
     *
     * Returns false on rejection so a caller that can act on it may; callers
     * that only need the diagnostic can ignore the result, since the logging has
     * already happened.
     *
     * @param host_field  Address inside the host shm shadow.
     * @param size        Bytes to publish.
     * @param what        Field name for the log line.
     */
    bool publish_field(const volatile void *host_field, size_t size, const char *what) {
        if (manager_.write_range_to_device(host_field, size) != 0) {
            LOG_ERROR(
                "%s: failed to publish %s to the device; it keeps its previous value", Module::kSubsystemName, what
            );
            return false;
        }
        return true;
    }

    /**
     * Ready buffers the drain path has retired without delivering them since the
     * last report_drain_drops(). The device published them, so their records are
     * inside `device_total`, but no collector ever saw them: a non-zero value
     * means part of the reconcile gap is host-side, and each one was logged as an
     * ERROR naming the buffer and the reason when it was retired.
     *
     * Counted in buffers, not records — the record count of a buffer the host
     * could not read is not recoverable.
     */
    uint64_t drain_dropped_buffers() const { return drain_dropped_buffers_.load(std::memory_order_relaxed); }

protected:
    // How long one ready queue's head entry may keep failing to be delivered
    // before it is retired. This is what bounds a host-side transport failure:
    // without it an entry the host can never read would hold its queue's head
    // forever, and quiesce() — which waits for every drain shard to report its
    // queues empty — could not complete. Only a broken path ever spends it, and
    // it is spent once per undeliverable entry.
    //
    // A duration rather than a sweep count because what it rides out is measured
    // in time: a sweep costs microseconds, so any count small enough to bound
    // teardown would expire long before a transport failure could clear.
    static constexpr std::chrono::milliseconds kStalledDrainEntryTimeout{1000};

    /**
     * Name what the drain path lost, so a reconcile gap is attributable instead
     * of anonymous. Derived::reconcile_counters() is the only caller: the read
     * consumes the count, which is what makes each run report its own rather
     * than the collector's cumulative total across a resident lifetime.
     */
    void report_drain_drops() {
        const uint64_t dropped = drain_dropped_buffers_.exchange(0, std::memory_order_relaxed);
        if (dropped == 0) return;
        LOG_ERROR(
            "%s reconcile: the host drain path retired %lu ready buffer(s) without delivering them. Their records are "
            "inside device_total but were never collected, so any silent_loss reported below is at least this far "
            "host-side; the per-buffer ERROR lines above name each buffer and why it was retired.",
            Derived::kSubsystemName, static_cast<unsigned long>(dropped)
        );
    }

    void bind_manager_memory_context() {
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
    }

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
            // A release that does not report success leaves memory held, so the
            // manager's paired occupancy keeps counting it rather than treating
            // an emptied mapping table as proof.
            if (int rc = free_cb(release_ptr); rc != 0) manager_.note_release_failed(release_ptr, rc);
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
                release_unregistered_buffer(dev_ptr);
                return nullptr;
            }
        } else if (copy_to_device_) {
            // Non-SVM: malloc + zero + push to device.
            host_ptr = std::malloc(size);
            if (host_ptr == nullptr) {
                LOG_ERROR("ProfilerBase::alloc_paired_buffer: host shadow alloc failed for %zu bytes", size);
                release_unregistered_buffer(dev_ptr);
                return nullptr;
            }
            std::memset(host_ptr, 0, size);
            int rc = copy_to_device_(dev_ptr, host_ptr, size);
            if (rc != 0) {
                LOG_ERROR("ProfilerBase::alloc_paired_buffer: copy_to_device failed: %d", rc);
                std::free(host_ptr);
                release_unregistered_buffer(dev_ptr);
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

    /**
     * Release a device pointer this allocation never registered, recording a
     * release that did not report success.
     *
     * Each of the three paths above returns before `register_mapping`, so the
     * pointer never reaches the manager's mapping table and the init rollback
     * guard can neither release it nor observe what happened to it: this is the
     * only place its outcome can be recorded. A context with no free callback
     * cannot release the pointer at all, which is the same conclusion — the
     * memory is still held — and is recorded the same way.
     */
    void release_unregistered_buffer(void *dev_ptr) {
        if (dev_ptr == nullptr) return;
        const int rc = free_cb_ ? free_cb_(dev_ptr) : -1;
        if (rc != 0) manager_.note_release_failed(dev_ptr, rc);
    }

    // -------------------------------------------------------------------------
    // Cross-run transport cut
    // -------------------------------------------------------------------------
    //
    // A finite, per-queue proof that one run's published buffers have all been
    // delivered and processed, usable while a successor keeps publishing. An
    // empty-sweep observation cannot do that: `found_any` is set by any entry on
    // any queue, so a busy successor keeps every sweep non-empty and the
    // quiescence ack never lands. The cut instead fixes a *target count* per
    // queue at one instant and then watches monotonic counters reach it.
    //
    // Every value below is written by exactly one thread: a queue's counters by
    // the drain owner that serves it (owner `q % shard_count_`), a shard's
    // processed counter by that collector thread. Nothing here is read or
    // written unless a run has been retained, so the five other profilers
    // pay one relaxed load per sweep and nothing else.

    static constexpr size_t kMaxCutSlots = 2;
    static constexpr size_t kMaxCutQueues = static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS);

    /**
     * A cut slot's lifecycle.
     *
     * `Published` is the only state in which a drain owner may touch the
     * per-queue arrays, and a slot reaches it only after they are initialized.
     * It returns to `Free` only once every owner has proved it is no longer
     * inside them, so the next arm cannot reinitialize an array under a reader
     * that observed the previous incarnation.
     */
    enum class CutState : int { Free = 0, Reserved, Published, Retiring };

    struct CutSlot {
        std::atomic<int> state{static_cast<int>(CutState::Free)};
        // Per queue, written once by that queue's owner when it captures the cut
        // at an entry boundary and read afterwards by that same owner. No other
        // thread reads `target`.
        std::array<uint64_t, kMaxCutQueues> target{};
        // 0 unarmed, 1 armed, 2 capture failed. Written by the queue's owner and
        // read by the writer, so the access is atomic even though the
        // writer is unique.
        std::array<std::atomic<uint8_t>, kMaxCutQueues> qstate{};
        // Per drain owner, published when every queue it serves has reached its
        // own target. The count of pushes at or before that instant is what the
        // collector side must catch up to.
        std::array<std::atomic<uint64_t>, Manager::kMaxCollectorShards> push_watermark{};
        std::array<std::atomic<uint8_t>, Manager::kMaxCollectorShards> stage1{};
    };

    /**
     * Arm a cut. Returns its slot index, or -1 when both slots are in use.
     *
     * `request_out` receives the capture request this arm published; every
     * later question about the cut is asked against that value, because an ack
     * for a *different* request says nothing about this capture.
     *
     * The caller must be holding the run's execution claim: the capture reads
     * `queue_tails[q]`, which is stable only while no producer is running.
     */
    int cut_arm(uint64_t *request_out) {
        if (request_out == nullptr) return -1;
        *request_out = 0;
        if (!simpler::dfx::runs::counter_headroom(cut_request_.load(std::memory_order_relaxed))) {
            note_counter_exhausted("cut request generation");
            return -1;
        }
        for (size_t slot = 0; slot < kMaxCutSlots; slot++) {
            int expected = static_cast<int>(CutState::Free);
            if (!cut_slots_[slot].state.compare_exchange_strong(
                    expected, static_cast<int>(CutState::Reserved), std::memory_order_acq_rel
                )) {
                continue;
            }
            // Reserved and not yet published, so no drain owner may read any of
            // this: initialize first, publish second.
            CutSlot &s = cut_slots_[slot];
            s.target.fill(0);
            for (size_t q = 0; q < kMaxCutQueues; q++)
                s.qstate[q].store(0, std::memory_order_relaxed);
            for (int i = 0; i < Manager::kMaxCollectorShards; i++) {
                s.push_watermark[i].store(0, std::memory_order_relaxed);
                s.stage1[i].store(0, std::memory_order_relaxed);
            }
            s.state.store(static_cast<int>(CutState::Published), std::memory_order_release);
            *request_out = cut_bump_request();
            return static_cast<int>(slot);
        }
        return -1;
    }

    /** Every drain owner has acknowledged a boundary pass at or after `request`. */
    bool cut_acked(uint64_t request) const {
        for (int i = 0; i < shard_count_; i++) {
            // `>=` and not `==`: an ack is monotonic, so an owner that has moved
            // past this request has certainly passed a boundary after it.
            if (cut_ack_[i].load(std::memory_order_acquire) < request) return false;
        }
        return true;
    }

    /** Block until `request` is acknowledged by every owner, or the budget runs out. */
    bool cut_wait_for_ack(uint64_t request, int timeout_ms) {
        if (!mgmt_running_.load(std::memory_order_acquire)) return cut_acked(request);
        std::unique_lock<std::mutex> lk(cut_mu_);
        return cut_cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [this, request] {
            return cut_acked(request);
        });
    }

    /** Stage 1: every owner reports all of its own queues at their targets. */
    bool cut_stage1_done(int slot) const {
        if (slot < 0 || static_cast<size_t>(slot) >= kMaxCutSlots) return false;
        const CutSlot &s = cut_slots_[static_cast<size_t>(slot)];
        if (s.state.load(std::memory_order_acquire) != static_cast<int>(CutState::Published)) return false;
        for (int i = 0; i < shard_count_; i++) {
            if (s.stage1[i].load(std::memory_order_acquire) == 0) return false;
        }
        return true;
    }

    /** Stage 2: every collector shard has processed up to its owner's watermark. */
    bool cut_stage2_done(int slot) const {
        if (!cut_stage1_done(slot)) return false;
        const CutSlot &s = cut_slots_[static_cast<size_t>(slot)];
        for (int i = 0; i < shard_count_; i++) {
            if (ring_processed_[i].load(std::memory_order_acquire) <
                s.push_watermark[i].load(std::memory_order_acquire)) {
                return false;
            }
        }
        return true;
    }

    /**
     * How many of this cut's queues could not be captured.
     *
     * False means *unknown*, not zero: until every drain owner has
     * acknowledged the capture request, a queue that has not been visited yet
     * is indistinguishable from one that succeeded, so reporting zero failures
     * would be the absence of a report dressed up as a clean one.
     */
    bool cut_failed_queues(int slot, uint64_t request, int *failed_out) const {
        if (failed_out == nullptr) return false;
        if (slot < 0 || static_cast<size_t>(slot) >= kMaxCutSlots) return false;
        const CutSlot &s = cut_slots_[static_cast<size_t>(slot)];
        if (s.state.load(std::memory_order_acquire) != static_cast<int>(CutState::Published)) return false;
        if (!cut_acked(request)) return false;
        int failed = 0;
        for (int q = 0; q < queue_count_ && static_cast<size_t>(q) < kMaxCutQueues; q++) {
            if (s.qstate[static_cast<size_t>(q)].load(std::memory_order_acquire) == 2) failed++;
        }
        *failed_out = failed;
        return true;
    }

    /**
     * Retire a cut and hand its slot back.
     *
     * Marking the slot non-published is not enough on its own: a drain owner
     * that already observed `Published` may still be inside the slot's arrays,
     * and the next arm would reinitialize them under it. So the retirement
     * publishes a fresh capture request *after* the state change and waits for
     * every owner to acknowledge a boundary pass that began after it — a pass
     * that, by the acquire on the request, must have observed `Retiring` and
     * therefore touched nothing. Earlier passes on that thread are finished by
     * program order.
     *
     * Returns false on timeout; the slot then stays retired for good rather
     * than being handed to a reader-visible reuse.
     */
    bool cut_release(int slot, int timeout_ms) {
        if (slot < 0 || static_cast<size_t>(slot) >= kMaxCutSlots) return true;
        CutSlot &s = cut_slots_[static_cast<size_t>(slot)];
        int expected = static_cast<int>(CutState::Published);
        if (!s.state.compare_exchange_strong(
                expected, static_cast<int>(CutState::Retiring), std::memory_order_acq_rel
            )) {
            if (expected == static_cast<int>(CutState::Reserved)) {
                // Reserved but never published: no drain owner can have seen it.
                s.state.store(static_cast<int>(CutState::Free), std::memory_order_release);
                return true;
            }
            // Already free, or being retired by somebody else — either way this
            // caller must not hand it back a second time.
            return expected == static_cast<int>(CutState::Free);
        }
        if (!mgmt_running_.load(std::memory_order_acquire)) {
            // No drain owner is running, so there is no reader to retire behind.
            s.state.store(static_cast<int>(CutState::Free), std::memory_order_release);
            return true;
        }
        const uint64_t request = cut_bump_request();
        if (!cut_wait_for_ack(request, timeout_ms)) {
            LOG_ERROR(
                "%s: cut slot %d could not be retired within %d ms; it is not reused", Module::kSubsystemName, slot,
                timeout_ms
            );
            return false;
        }
        s.state.store(static_cast<int>(CutState::Free), std::memory_order_release);
        return true;
    }

    /**
     * Bound the entries one queue may consume before the sweep rotates.
     *
     * Zero keeps today's behaviour — drain each queue until it reports empty —
     * which starves a quiet queue while a busy sibling is served, and therefore
     * starves that queue's stage 1. Retaining runs sets a finite quantum; nothing
     * else does.
     */
    void set_drain_quantum(int quantum) { drain_quantum_.store(quantum, std::memory_order_relaxed); }

    /**
     * Arm or disarm the per-entry transport counters retention needs.
     *
     * Armed for the whole time runs are retained, not per cut: a counter that started
     * counting at the first arm would have missed every entry consumed before
     * it, and stage 1 compares a target captured from that same counter. While
     * disarmed — which is every profiler that never retains a run — the drain
     * loop pays one relaxed load per queue visit and the collector one per
     * buffer, and no atomic is written.
     */
    void set_run_counters(bool on) { run_counters_on_.store(on, std::memory_order_release); }

    /** A transport counter ran out of headroom; no cut can be trusted after this. */
    bool cut_counters_exhausted() const { return cut_counter_exhausted_.load(std::memory_order_acquire); }

    /**
     * Publish a reference-release request and wait for every collector shard.
     *
     * The shard loads this epoch *before* refreshing its own view of the
     * retained-run table, so an ack can never describe a view taken before
     * the caller marked an epoch non-admitting. Returns false on timeout, and a
     * false return is never permission to free: the caller quarantines.
     */
    bool request_run_reference_release(int timeout_ms) {
        // One requester at a time. Two overlapping requests would each wait for
        // their own epoch value while a shard, which only ever adopts the
        // newest, could skip the older one entirely — so the older waiter would
        // time out and quarantine a bucket that was in fact released.
        std::lock_guard<std::mutex> lk(control_mu_);
        if (!simpler::dfx::runs::counter_headroom(control_epoch_.load(std::memory_order_relaxed))) {
            note_counter_exhausted("retained-run control epoch");
            return false;
        }
        const uint64_t epoch = control_epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
        manager_.notify_ready_waiters();
        std::unique_lock<std::mutex> wait_lk(cut_mu_);
        return cut_cv_.wait_for(wait_lk, std::chrono::milliseconds(timeout_ms), [this, epoch] {
            for (int i = 0; i < shard_count_; i++) {
                // `>=` and not `==`: an ack is monotonic, and a shard that has
                // already moved past this epoch has certainly passed it.
                if (control_acked_[i].load(std::memory_order_acquire) < epoch) return false;
            }
            return true;
        });
    }

private:
    /** Publish a capture request and return it. Acks are compared against it. */
    uint64_t cut_bump_request() { return cut_request_.fetch_add(1, std::memory_order_acq_rel) + 1; }

    /**
     * A transport counter has come within `kCounterMargin` of wrapping.
     *
     * Reported once and sticky: a wrapped counter makes every target
     * comparison meaningless, so the collector refuses rather than publishing a
     * cut it cannot justify. Waiters are woken because the refusal is what they
     * are waiting to learn.
     */
    void note_counter_exhausted(const char *what) {
        bool expected = false;
        if (!cut_counter_exhausted_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;
        LOG_ERROR("%s: %s counter is out of headroom; no further cut is trustworthy", Module::kSubsystemName, what);
        {
            std::lock_guard<std::mutex> lk(cut_mu_);
            cut_cv_.notify_all();
        }
        notify_transport_progress(0);
    }

    /**
     * Wake anything waiting on an ack.
     *
     * The mutex is taken after the ack store is already visible, so a waiter
     * either evaluates its predicate afterwards and sees the ack, or is already
     * blocked and is woken here. There is no window in between.
     */
    void cut_notify_ack() {
        std::lock_guard<std::mutex> lk(cut_mu_);
        cut_cv_.notify_all();
    }

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

        // When each of this shard's queues first failed to deliver its head
        // entry, or a default-constructed time point while it is delivering
        // normally. Per queue so one stalled lane does not spend a sibling's
        // budget.
        std::vector<std::chrono::steady_clock::time_point> stalled_since(static_cast<size_t>(queue_count_));

        while (mgmt_running_.load(std::memory_order_relaxed)) {
            // An ack covers only a sweep that starts after this epoch is observed.
            const uint64_t requested = drain_quiesce_epoch_.load(std::memory_order_acquire);
            // `found_any` gates quiescence — it means a queue still holds an entry
            // that must be delivered. `retired_or_delivered` gates the idle
            // backoff, and they differ only while an entry is being retried: a
            // stalled queue is not idle, but it is also not making progress, and
            // polling it flat out for the whole budget would burn a core.
            bool found_any = false;
            bool retired_or_delivered = false;
            for (int q = queue_start; q < queue_count_; q += queue_stride) {
                // Entry boundary: nothing of this queue's head is half-processed
                // here, so the capture and its stage-1 check see a
                // consistent (head, consumed, queue contents) triple. Checked on
                // every visit, including a queue that turns out to be empty and
                // one whose last outcome was a retry, so a busy sibling can never
                // hide a pending request.
                run_drain_boundary(header, queue_start, queue_stride);
                ReadyEntry entry;
                int served = 0;
                const int quantum = drain_quantum_.load(std::memory_order_relaxed);
                while (Alg::try_peek_aicpu_entry(manager_, header, q, entry, true)) {
                    // A null free_queue is the "nothing to retry" sentinel;
                    // process_entry only writes this on a short top-up.
                    EntrySite<Module> short_site{};
                    auto &since = stalled_since[static_cast<size_t>(q)];
                    const bool exhausted = entry_retries_exhausted(since);
                    const EntryOutcome outcome = Alg::process_entry(manager_, header, q, entry, &short_site, exhausted);
                    if (outcome == EntryOutcome::kRetry) {
                        if (since == std::chrono::steady_clock::time_point{}) {
                            since = std::chrono::steady_clock::now();
                        }
                        // Within the budget the entry is still deliverable, so the
                        // queue counts as live and quiesce() must wait for it. Past
                        // the budget only a failing acknowledgement write can
                        // produce a retry — the device link is gone and cannot
                        // recover, so reporting the queue live would hang quiesce()
                        // instead of letting the logged write failure be the signal.
                        if (!exhausted) {
                            found_any = true;
                        }
                        // The entry still sits at the head either way, so move on
                        // to the next queue rather than spinning on it.
                        break;
                    }
                    found_any = true;
                    retired_or_delivered = true;
                    since = std::chrono::steady_clock::time_point{};
                    if (outcome == EntryOutcome::kDropped) {
                        drain_dropped_buffers_.fetch_add(1, std::memory_order_relaxed);
                        note_buffer_retired(q);
                    } else {
                        note_buffer_delivered(q, queue_start);
                    }
                    if (short_site.free_queue != nullptr) {
                        record_short_site(short_sites, q, short_site);
                    }
                    // Rotate after a bounded number of entries so a sustained
                    // producer on this queue cannot starve a sibling — and with
                    // it, that sibling's cut. Zero means today's drain-to-empty.
                    if (quantum > 0 && ++served >= quantum) break;
                }
            }
            run_drain_boundary(header, queue_start, queue_stride);
            if (retired_or_delivered) {
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
                if (drain_acked_[queue_start].load(std::memory_order_relaxed) != requested) {
                    drain_acked_[queue_start].store(requested, std::memory_order_release);
                }
            }

            if (!retired_or_delivered) {
                if (idle_busy_polls < kIdleBusyPollLoops) {
                    idle_busy_polls++;
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            }
        }

        for (int q = queue_start; q < queue_count_; q += queue_stride) {
            ReadyEntry entry;
            auto since = std::chrono::steady_clock::time_point{};
            while (Alg::try_peek_aicpu_entry(manager_, header, q, entry, true)) {
                const bool exhausted = entry_retries_exhausted(since);
                const EntryOutcome outcome = Alg::process_entry(manager_, header, q, entry, nullptr, exhausted);
                if (outcome == EntryOutcome::kRetry) {
                    // Past the budget the acknowledgement write is what is
                    // failing; leave the queue alone rather than retrying a dead
                    // device link until the pass never ends.
                    if (exhausted) break;
                    if (since == std::chrono::steady_clock::time_point{}) {
                        since = std::chrono::steady_clock::now();
                    }
                    // Teardown path, so a sleep is permitted here (codestyle.md
                    // rule 5): no task's latency passes through it, and it keeps a
                    // dead device link from pinning a core for the whole budget.
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                    continue;
                }
                since = std::chrono::steady_clock::time_point{};
                if (outcome == EntryOutcome::kDropped) {
                    drain_dropped_buffers_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Transport cut bookkeeping, all single-writer
    // -------------------------------------------------------------------------

    /** Count one entry this owner delivered to its collector shard. */
    void note_buffer_delivered(int q, int owner) {
        if (!run_counters_on_.load(std::memory_order_relaxed)) return;
        if (static_cast<size_t>(q) >= kMaxCutQueues || owner < 0 || owner >= Manager::kMaxCollectorShards) return;
        const bool ok = simpler::dfx::runs::checked_increment(consumed_total_[static_cast<size_t>(q)]) &&
                        simpler::dfx::runs::checked_increment(pushed_total_[static_cast<size_t>(owner)]);
        if (!ok) note_counter_exhausted("per-queue transport");
    }

    /**
     * Count one entry this owner consumed without delivering it.
     *
     * A retired entry is consumed as far as the cut is concerned — its slot is
     * gone and no collector will ever see it — so the target stays reachable.
     * The loss itself is already reported by `drain_dropped_buffers_`.
     */
    void note_buffer_retired(int q) {
        if (!run_counters_on_.load(std::memory_order_relaxed)) return;
        if (static_cast<size_t>(q) >= kMaxCutQueues) return;
        if (!simpler::dfx::runs::checked_increment(consumed_total_[static_cast<size_t>(q)])) {
            note_counter_exhausted("per-queue transport");
        }
    }

    /**
     * The only place a cut is captured or advanced.
     *
     * Called at entry boundaries only, which is what makes the triple it reads
     * consistent. Arming refreshes the tail narrowly through the same
     * per-word call `try_peek_aicpu_entry` uses, and issues it from the word's
     * sole owner so the host shadow keeps one writer.
     *
     * A slot is touched only in `Published`, which is published after its
     * arrays are initialized and withdrawn before they are reinitialized — and
     * the ack this pass writes at the end is what proves to a retiring cut that
     * this owner is no longer inside them.
     */
    void run_drain_boundary(DataHeader *header, int queue_start, int queue_stride) {
        if (!run_counters_on_.load(std::memory_order_relaxed)) return;
        const uint64_t request = cut_request_.load(std::memory_order_acquire);
        const bool capture_pending = cut_ack_[queue_start].load(std::memory_order_relaxed) < request;
        bool progressed = false;

        for (size_t slot = 0; slot < kMaxCutSlots; slot++) {
            CutSlot &s = cut_slots_[slot];
            if (s.state.load(std::memory_order_acquire) != static_cast<int>(CutState::Published)) continue;
            for (int q = queue_start; q < queue_count_ && static_cast<size_t>(q) < kMaxCutQueues; q += queue_stride) {
                if (s.qstate[static_cast<size_t>(q)].load(std::memory_order_relaxed) != 0) continue;
                uint32_t tail = 0;
                uint32_t head = 0;
                if (!capture_run_queue(header, q, &head, &tail)) {
                    // CaptureFailed: this queue's stage 1 is unknown and is
                    // never satisfied by default.
                    s.qstate[static_cast<size_t>(q)].store(2, std::memory_order_release);
                    continue;
                }
                const uint32_t outstanding = (tail + Module::kReadyQueueSize - head) % Module::kReadyQueueSize;
                s.target[static_cast<size_t>(q)] =
                    consumed_total_[static_cast<size_t>(q)].load(std::memory_order_relaxed) + outstanding;
                s.qstate[static_cast<size_t>(q)].store(1, std::memory_order_release);
            }
            if (s.stage1[queue_start].load(std::memory_order_relaxed) != 0) continue;
            bool all_reached = true;
            for (int q = queue_start; q < queue_count_ && static_cast<size_t>(q) < kMaxCutQueues; q += queue_stride) {
                const uint8_t state = s.qstate[static_cast<size_t>(q)].load(std::memory_order_relaxed);
                if (state == 0) {
                    all_reached = false;  // not captured yet
                    break;
                }
                if (state == 2) continue;  // failed queues cannot be waited for
                if (consumed_total_[static_cast<size_t>(q)].load(std::memory_order_relaxed) <
                    s.target[static_cast<size_t>(q)]) {
                    all_reached = false;
                    break;
                }
            }
            if (all_reached) {
                s.push_watermark[queue_start].store(
                    pushed_total_[static_cast<size_t>(queue_start)].load(std::memory_order_relaxed),
                    std::memory_order_release
                );
                s.stage1[queue_start].store(1, std::memory_order_release);
                progressed = true;
            }
        }
        if (capture_pending) {
            cut_ack_[queue_start].store(request, std::memory_order_release);
            cut_notify_ack();
        }
        // Stage 1 is what a publisher waits for, so it is woken by the
        // transition rather than by a timer.
        if (progressed) notify_transport_progress(0);
    }

    /** Narrow, owner-issued refresh of one queue's cursors. */
    bool capture_run_queue(DataHeader *header, int q, uint32_t *head_out, uint32_t *tail_out) {
        if (header == nullptr) return false;
        if (manager_.read_range_from_device(&header->queue_heads[q], sizeof(header->queue_heads[q])) != 0 ||
            manager_.read_range_from_device(&header->queue_tails[q], sizeof(header->queue_tails[q])) != 0) {
            LOG_ERROR("%s: cut could not refresh ready_queue cursors for thread %d", Module::kSubsystemName, q);
            return false;
        }
        rmb();
        const uint32_t head = header->queue_heads[q];
        const uint32_t tail = header->queue_tails[q];
        if (head >= Module::kReadyQueueSize || tail >= Module::kReadyQueueSize) return false;
        *head_out = head;
        *tail_out = tail;
        return true;
    }

    // A stall that has not started yet is never exhausted; one that has is
    // exhausted once it has outlived kStalledDrainEntryTimeout.
    static bool entry_retries_exhausted(std::chrono::steady_clock::time_point stalled_since) {
        if (stalled_since == std::chrono::steady_clock::time_point{}) return false;
        return std::chrono::steady_clock::now() - stalled_since >= kStalledDrainEntryTimeout;
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

    bool quiesce_pending(int shard_index) const {
        return collect_quiesce_epoch_.load(std::memory_order_acquire) !=
               collect_acked_[shard_index].load(std::memory_order_relaxed);
    }

    /**
     * This shard owes an acknowledgement for the retained-run table.
     *
     * Part of the ready-ring wait predicate, and level-triggered like the
     * quiescence term: it stays true until the shard stores its ack at the top
     * of the loop, after the refresh. Notifying the ring is not enough on its
     * own — `notify_ready_waiters` wakes the consumer but advances no ready
     * shard's `state_epoch`, so a consumer asleep on an unchanging empty ring
     * would re-test a predicate that knows nothing about control, find it
     * false, and sleep out the rest of its 100 ms tick. A request that lands
     * between the epoch load at the top of the loop and the wait is the same
     * case with the same answer: the condition is in the predicate, so the wait
     * returns at once instead of timing out. `run_begin` blocks on this
     * acknowledgement before a device launch, so that tick would be paid by
     * every run.
     */
    bool control_pending(int shard_index) const {
        return control_epoch_.load(std::memory_order_acquire) !=
               control_acked_[shard_index].load(std::memory_order_relaxed);
    }

    /**
     * Main collector loop. Blocks on one manager ready-queue shard. Ready
     * buffers and lifecycle control requests wake it immediately; the 100 ms
     * cv-wait tick is a fallback for missed data-path notifications and idle
     * bookkeeping. On each hit it dispatches the buffer to Derived via
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
            // Reference-release handshake, at the top of every iteration and
            // under load — not only when this shard's ring runs dry. The control
            // epoch is read *before* the snapshot refresh so this ack can never
            // describe a view taken before the collector marked a run
            // non-admitting, and it is emitted while this shard holds no bucket
            // reference.
            {
                const uint64_t ctrl = control_epoch_.load(std::memory_order_acquire);
                if (ctrl != control_acked_[shard_index].load(std::memory_order_relaxed)) {
                    refresh_retained_run_view(shard_index, 0);
                    control_acked_[shard_index].store(ctrl, std::memory_order_release);
                    cut_notify_ack();
                }
            }
            // Refreshed and acked above, so the control term of the wait
            // predicate below is false again by the time the wait is entered
            // unless a newer request has already arrived.
            ReadyBufferInfo info;
            if (manager_.wait_pop_ready(info, wait_tick, shard_index, [this, shard_index] {
                    return execution_complete_.load(std::memory_order_acquire) || quiesce_pending(shard_index) ||
                           control_pending(shard_index);
                })) {
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
            // Phase two of the quiescence handshake. A false wait result means
            // no ready buffer was available after either a timeout or a control
            // wake. For a pending quiesce, mgmt has already reported its sweep
            // done for this epoch, so nothing further can arrive. Placed above
            // the has_seen_buffer guard below: a shard that never received a
            // buffer is a valid run shape and still has to report, or quiesce()
            // would wait on it forever.
            {
                const uint64_t requested = collect_quiesce_epoch_.load(std::memory_order_acquire);
                if (quiesce_pending(shard_index)) {
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
        // After the copy, never before: stage 2 is what licenses
        // moving this shard's records, so the count must not run ahead of them.
        if (run_counters_on_.load(std::memory_order_relaxed) && shard_index >= 0 &&
            shard_index < Manager::kMaxCollectorShards) {
            const uint64_t processed =
                ring_processed_[static_cast<size_t>(shard_index)].fetch_add(1, std::memory_order_release) + 1;
            if (!simpler::dfx::runs::counter_headroom(processed, 0)) {
                note_counter_exhausted("per-shard processed");
            }
            // Exactly the instant this shard satisfies an armed cut's stage 2,
            // so the publisher is woken by the event and not by a timer.
            if (cut_watermark_reached(shard_index, processed)) notify_transport_progress(0);
        }
        if constexpr (Module::kBufferKinds > 1) {
            (void)manager_.notify_copy_done(info.dev_buffer_ptr, Module::kind_of(info), shard_index);
        } else {
            (void)manager_.notify_copy_done(info.dev_buffer_ptr, 0, shard_index);
        }
    }

    /** True at the step on which this shard reaches an armed cut's watermark. */
    bool cut_watermark_reached(int shard_index, uint64_t processed) const {
        for (size_t slot = 0; slot < kMaxCutSlots; slot++) {
            const CutSlot &s = cut_slots_[slot];
            if (s.state.load(std::memory_order_acquire) != static_cast<int>(CutState::Published)) continue;
            if (s.stage1[shard_index].load(std::memory_order_acquire) == 0) continue;
            if (s.push_watermark[shard_index].load(std::memory_order_acquire) == processed) return true;
        }
        return false;
    }

    std::vector<std::thread> mgmt_drain_threads_;
    std::thread mgmt_replenish_thread_;
    std::atomic<bool> mgmt_running_{false};

    // Written by every drain shard, read once per run by reconcile_counters().
    std::atomic<uint64_t> drain_dropped_buffers_{0};

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

    // Cross-run cut state. Inert while no run has been retained in the
    // counters: the drain loop pays one relaxed load per queue visit and the
    // collector one per buffer, and no other profiler arms them.
    std::array<CutSlot, kMaxCutSlots> cut_slots_{};
    std::atomic<uint64_t> cut_request_{0};
    std::array<std::atomic<uint64_t>, Manager::kMaxCollectorShards> cut_ack_{};
    std::array<std::atomic<uint64_t>, kMaxCutQueues> consumed_total_{};
    std::array<std::atomic<uint64_t>, Manager::kMaxCollectorShards> pushed_total_{};
    std::array<std::atomic<uint64_t>, Manager::kMaxCollectorShards> ring_processed_{};
    std::atomic<int> drain_quantum_{0};
    std::atomic<bool> run_counters_on_{false};
    std::atomic<bool> cut_counter_exhausted_{false};
    std::atomic<uint64_t> control_epoch_{0};
    std::array<std::atomic<uint64_t>, Manager::kMaxCollectorShards> control_acked_{};
    std::mutex control_mu_;
    // Wakeups for the two ack handshakes — capture/retirement and reference
    // release. Both are waited on by a caller and satisfied by a drain owner or
    // a collector shard, so neither side spins.
    std::mutex cut_mu_;
    std::condition_variable cut_cv_;

    /**
     * Optional Derived hook: refresh that shard's private view of the
     * epoch table. A collector that defines no such method gets the `long`
     * overload and no behaviour change — the same overload-rank idiom
     * `refresh_replenish_metadata` already uses here.
     */
    template <typename D = Derived>
    auto refresh_retained_run_view(int shard_index, int)
        -> decltype(static_cast<D *>(this)->refresh_retained_run_view(shard_index), void()) {
        static_cast<D *>(this)->refresh_retained_run_view(shard_index);
    }
    template <typename D = Derived>
    void refresh_retained_run_view(int, long) {}

    /**
     * Optional Derived hook: transport progress the writer is
     * waiting on has happened. Called only at the exact transitions — a
     * stage-1 publication, a shard reaching its watermark, a counter refusal —
     * so the publisher needs no periodic poll to notice them.
     */
    template <typename D = Derived>
    auto notify_transport_progress(int) -> decltype(static_cast<D *>(this)->note_transport_progress(), void()) {
        static_cast<D *>(this)->note_transport_progress();
    }
    template <typename D = Derived>
    void notify_transport_progress(long) {}
};

}  // namespace profiling_common
