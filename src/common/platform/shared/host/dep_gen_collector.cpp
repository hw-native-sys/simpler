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
 * @file dep_gen_collector.cpp
 * @brief Host-side dep_gen collector. The mgmt-thread + buffer-pool machinery
 *        lives in profiling_common::BufferPoolManager parameterized by
 *        DepGenModule (host/dep_gen_collector.h); this file owns the
 *        per-buffer on_buffer_collected callback (in-memory append) and the
 *        device-side cross-check. Records stay in ``records_by_run_`` and are
 *        consumed directly by the host replay — no on-disk submit_trace.bin
 *        intermediary.
 *
 * Non-SVM platforms route device↔host transfers through profiling_copy.h.
 * Each DepGenBuffer's contents are pulled from device on demand inside
 * ProfilerAlgorithms::process_entry, so on_buffer_collected can read `count`
 * and `records[]` directly off the host shadow.
 */

#include "host/dep_gen_collector.h"

#include <cassert>
#include <cstring>
#include <unordered_set>

#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host/profiling_copy.h"
#include "../../../worker/runtime_c_api.h"

DepGenCollector::~DepGenCollector() {
    // The writer outlives the collector threads, so it is joined here too: a
    // joinable thread left behind at destruction terminates the process.
    stop_writer();
    stop();
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

int DepGenCollector::init(
    int num_threads, const DepGenAllocCallback &alloc_cb, DepGenRegisterCallback register_cb,
    const DepGenFreeCallback &free_cb, int device_id
) {
    if (initialized_) {
        // Already holding this run's device resources. They are not per-run:
        // configuration arrives via begin_run() and the layout is fixed at
        // compile time, so there is nothing here left to re-apply.
        return 0;
    }
    if (num_threads <= 0 || num_threads > PLATFORM_MAX_AICPU_THREADS || alloc_cb == nullptr || free_cb == nullptr) {
        LOG_ERROR(
            "DepGenCollector::init: invalid arguments (num_threads=%d, valid range: 1-%d)", num_threads,
            PLATFORM_MAX_AICPU_THREADS
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // Must precede the recycled-lane seeding below: push_recycled() folds its
    // shard argument modulo the manager's shard count.
    set_aicpu_thread_num(num_threads);

    num_threads_ = num_threads;
    total_collected_ = 0;
    records_by_run_.clear();

    // Stash callbacks on the base up-front so alloc_paired_buffer sees
    // consistent values during init. shm_host_ stays nullptr until the shm
    // allocation succeeds — start(tf) gates on shm_host_.
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_or_null(), profiling_copy_from_device_or_null(),
        /*shm_dev=*/nullptr, /*shm_host=*/nullptr, /*shm_size=*/0, device_id
    );

    // RAII rollback: any early return after this point releases the shm
    // region + every DepGenBuffer (device pointer + malloc'd host shadow)
    // tracked by the manager. `guard.commit()` runs on the success path
    // before the trailing return 0.
    profiling_common::InitRollbackGuard<decltype(manager_)> guard(manager_, free_cb);

    // ---- Allocate shared header + buffer-state region ----
    // dep_gen is single-instance: just one DepGenBufferState after the header.
    const int num_instances = 1;
    size_t shm_size = calc_dep_gen_shm_size(num_instances);
    void *shm_host_local = nullptr;
    void *shm_dev_local = alloc_paired_buffer(shm_size, &shm_host_local);
    if (shm_dev_local == nullptr) {
        LOG_ERROR("DepGenCollector: failed to allocate dep_gen shared memory (%zu bytes)", shm_size);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    std::memset(shm_host_local, 0, shm_size);
    DepGenDataHeader *hdr = get_dep_gen_header(shm_host_local);
    hdr->num_instances = static_cast<uint32_t>(num_instances);

    // ---- Allocate DepGenBuffers, populate free_queue + recycled pool ----
    const size_t buf_size = sizeof(DepGenBuffer);
    DepGenBufferState *state = get_dep_gen_buffer_state(shm_host_local, 0);

    const int owner_shard = (num_threads > 0) ? (num_threads - 1) : 0;
    for (int b = 0; b < PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE; b++) {
        void *host_ptr = nullptr;
        void *dev_ptr = alloc_paired_buffer(buf_size, &host_ptr);
        if (dev_ptr == nullptr) {
            LOG_ERROR("DepGenCollector: failed to allocate DepGenBuffer b=%d", b);
            return PTO_RUNTIME_ERR_INTERNAL;
        }

        if (b < PLATFORM_DEP_GEN_SLOT_COUNT) {
            // First N buffers go directly into the (single) instance's free_queue.
            uint32_t tail = state->free_queue.tail;
            assert(tail - state->free_queue.head < PLATFORM_DEP_GEN_SLOT_COUNT && "free_queue overflow on init");
            state->free_queue.buffer_ptrs[tail % PLATFORM_DEP_GEN_SLOT_COUNT] = reinterpret_cast<uint64_t>(dev_ptr);
            wmb();
            state->free_queue.tail = tail + 1;
            wmb();
        } else {
            if (!manager_.push_recycled(0, dev_ptr, owner_shard)) {
                (void)manager_.retire_unqueued_buffer(0, dev_ptr, owner_shard);
            }
        }
    }

    // Push the entire initialized shm region (header + BufferState +
    // free_queue contents) to device.
    profiling_copy_to_device(shm_dev_local, shm_host_local, shm_size);

    LOG_INFO(
        "DepGen collector initialized: %d threads, SHM=0x%lx (records held in memory until replay)", num_threads,
        reinterpret_cast<unsigned long>(shm_dev_local)
    );
    guard.commit();
    // Publish members + memory context only after the rollback guard is
    // disarmed; initialized_ is published last. set_memory_context copy-assigns
    // std::function members and can throw std::bad_alloc, so keeping it (and the
    // initialized_ store) before commit would let the guard free the shm/buffers
    // on unwind while initialized_ is already true — finalize() would then skip
    // its !initialized_ early-return and double-free what the guard released.
    // start(tf) gates on shm_host_ (published by set_memory_context), so this is
    // the moment the collector becomes startable.
    shm_dev_ = shm_dev_local;
    shm_size_ = shm_size;
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_or_null(), profiling_copy_from_device_or_null(),
        shm_dev_local, shm_host_local, shm_size, device_id
    );
    initialized_ = true;
    return 0;
}

// ---------------------------------------------------------------------------
// Record accumulation (in-memory — no disk hop)
// ---------------------------------------------------------------------------

bool DepGenCollector::begin_run() {
    {
        std::scoped_lock lock(records_mutex_);
        records_by_run_.clear();
    }
    total_collected_ = 0;
    reset_unpublished_ = false;
    // A sealed export owns its own records, so this releases only what this
    // collector still holds for a run that never sealed. The counters reset
    // with it, under the lock that owns them.
    size_t charged = 0;
    {
        std::scoped_lock lock(records_mutex_);
        host_clamped_ = false;
        refused_records_ = 0;
        foreign_epoch_records_ = 0;
        charged = retained_records_.release();
    }
    if (charged > 0) credit_record_block(charged);

    // Nothing on the device yet before the first init(); its memset covers this.
    if (shm_host_ == nullptr) return true;

    // The device's record counters are monotonic by contract (see dep_gen.h), so
    // they carry the previous run's totals into this one's reconcile. Zeroing
    // them is safe here and only here: the caller has quiesced the device, so
    // the AICPU is not writing them concurrently.
    DepGenBufferState *state = get_dep_gen_buffer_state(shm_host_, 0);
    state->total_record_count = 0;
    state->dropped_record_count = 0;
    state->total_overflow_record_count = 0;
    wmb();
    // Narrow write-backs, not the region: a bulk push would clobber the
    // device-owned fields next to these (current_buf_ptr, free_queue.head).
    // The results decide the run: a reset that never reached the device leaves
    // this run reconciling against the previous run's totals, so the identity
    // it reports is about two runs rather than one.
    bool published = publish_field(&state->total_record_count, sizeof(state->total_record_count), "total_record_count");
    published &=
        publish_field(&state->dropped_record_count, sizeof(state->dropped_record_count), "dropped_record_count");
    published &= publish_field(
        &state->total_overflow_record_count, sizeof(state->total_overflow_record_count), "total_overflow_record_count"
    );
    if (fail_counter_reset_) published = false;
    if (!published) {
        // Sticky for this run in both modes. Retained mode refuses the
        // admission that follows; the default path keeps its return code and
        // withholds the file instead.
        reset_unpublished_ = true;
        LOG_ERROR("dep_gen: this run's device counter reset was not published — no graph will be produced for it");
    }
    return published;
}

void DepGenCollector::append_buffer_records(const void *buf_host_ptr) {
    const DepGenBuffer *buf = reinterpret_cast<const DepGenBuffer *>(buf_host_ptr);
    bool clamped = false;
    uint32_t n = buf->count;
    if (n > static_cast<uint32_t>(PLATFORM_DEP_GEN_RECORDS_PER_BUFFER)) {
        // The excess is unreadable and is not counted anywhere on the device,
        // so the identity below would still balance over a trace missing
        // records. Recording that the clamp fired is what makes the graph
        // unpublishable instead of quietly short.
        LOG_ERROR(
            "dep_gen: a buffer claimed %u records, above the %d its slot holds — this run's graph is incomplete", n,
            PLATFORM_DEP_GEN_RECORDS_PER_BUFFER
        );
        n = static_cast<uint32_t>(PLATFORM_DEP_GEN_RECORDS_PER_BUFFER);
        clamped = true;
    }
    if (n == 0 && !clamped) return;

    // Read the identity before copying: it decides which run's graph these
    // records join. The device buffer goes back to the pool after this and a
    // later run re-stamps it, so nothing may consult it again.
    const uint64_t run_epoch = buf->run_epoch;

    std::scoped_lock lock(records_mutex_);
    // Every field below is this lock's, including the clamp flag the run
    // boundary reads: they are written on collector threads and read by the
    // boundary, so one owner rather than a plain field two locks reach.
    if (clamped) host_clamped_ = true;
    if (n == 0) return;
    if (retain_across_runs_ && retained_ready_) {
        // Grouping by whatever epoch a buffer carries is not identity: an
        // epoch this collector never admitted names no run, and filing it
        // under a fresh key would make a stray buffer look like a graph.
        if (!retained_epoch_open_ || run_epoch != retained_epoch_) {
            foreign_epoch_records_ += n;
            LOG_ERROR(
                "dep_gen: %u records arrived stamped 0x%llx, which is not the admitted run — this run's graph is "
                "untrustworthy",
                n, static_cast<unsigned long long>(run_epoch)
            );
            total_collected_ += n;
            return;
        }
        for (uint32_t i = 0; i < n; i++) {
            const bool stored = retained_records_.append(
                buf->records[i],
                [this](size_t bytes) {
                    return charge_record_block(bytes);
                },
                [this](size_t bytes) {
                    credit_record_block(bytes);
                }
            );
            if (!stored) {
                // Counted as received and not retained. A graph missing edges
                // is not a graph, so this run publishes nothing — but the
                // receive path is never blocked and nothing is allocated here.
                refused_records_ += static_cast<uint64_t>(n - i);
                total_collected_ += n;
                return;
            }
        }
        total_collected_ += n;
        return;
    }
    std::vector<DepGenRecord> &run_records = records_by_run_[run_epoch];
    run_records.insert(run_records.end(), buf->records, buf->records + n);
    total_collected_ += n;
}

// ---------------------------------------------------------------------------
// ProfilerBase callback
// ---------------------------------------------------------------------------

void DepGenCollector::on_buffer_collected(const DepGenReadyBufferInfo &info) {
    append_buffer_records(info.host_buffer_ptr);
}

// ---------------------------------------------------------------------------
// reconcile_counters
// ---------------------------------------------------------------------------

bool DepGenCollector::reconcile_counters() { return reconcile_report().clean; }

simpler::dfx::dep_gen_runs::ReconcileReport DepGenCollector::reconcile_report() {
    simpler::dfx::dep_gen_runs::ReconcileReport out;
    if (shm_host_ == nullptr) return out;
    report_drain_drops();

    // mgmt thread is stopped by the caller; pull the latest BufferState
    // (current_buf_ptr, total/dropped counters) from device so the
    // cross-check sees post-stop() values. The result decides the run: a failed
    // copy leaves the host shadow holding the zeroes begin_run() wrote, and
    // the identity below would balance over a run that collected nothing.
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        const int rc = profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
        if (rc != 0 || fail_region_read_) {
            LOG_ERROR("dep_gen reconcile: the shared region read failed (%d) — this run's counts are untrusted", rc);
            out.region_read_failed = true;
            return out;
        }
    } else if (fail_region_read_) {
        LOG_ERROR("dep_gen reconcile: the shared region read failed — this run's counts are untrusted");
        out.region_read_failed = true;
        return out;
    }
    rmb();

    bool clean = true;

    DepGenBufferState *state = dep_gen_state(0);
    uint64_t buf_dev = state->current_buf_ptr;
    if (buf_dev != 0) {
        void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_dev));
        if (host_ptr == nullptr) {
            // Skipping the check is not passing it: nothing established whether
            // that buffer still holds records.
            LOG_ERROR(
                "dep_gen reconcile: the buffer the device still holds (0x%lx) has no host mapping — nothing could be "
                "checked",
                static_cast<unsigned long>(buf_dev)
            );
            out.buffer_unresolved = true;
            clean = false;
        } else {
            const int rc =
                profiling_copy_from_device(host_ptr, reinterpret_cast<void *>(buf_dev), sizeof(DepGenBuffer));
            if (rc != 0) {
                LOG_ERROR(
                    "dep_gen reconcile: reading the buffer the device still holds (0x%lx) failed (%d)",
                    static_cast<unsigned long>(buf_dev), rc
                );
                out.buffer_read_failed = true;
                clean = false;
            } else {
                rmb();
                uint32_t count = reinterpret_cast<const DepGenBuffer *>(host_ptr)->count;
                if (count != 0) {
                    LOG_ERROR(
                        "dep_gen reconcile: un-flushed buffer (current_buf_ptr=0x%lx, count=%u) — device flush failed",
                        static_cast<unsigned long>(buf_dev), count
                    );
                    out.unflushed_records = true;
                    clean = false;
                }
            }
        }
    }

    uint64_t total_device = state->total_record_count;
    uint64_t dropped_device = state->dropped_record_count;
    uint64_t overflow_device = state->total_overflow_record_count;

    out.total_device = total_device;
    out.dropped_device = dropped_device;
    out.overflow_device = overflow_device;
    out.collected_host = total_collected_;
    {
        // Written on collector threads; the drain is quiesced by the caller, so
        // this is a consistent read rather than a racing one.
        std::scoped_lock lock(records_mutex_);
        out.host_clamped = host_clamped_;
        out.identity_foreign = foreign_epoch_records_ > 0 || refused_records_ > 0;
    }
    out.reset_unpublished = reset_unpublished_;
    if (out.host_clamped || out.identity_foreign || out.reset_unpublished) clean = false;

    // The saturation sentinel: the producer clamps at UINT32_MAX rather than
    // wrapping, so a counter reading it means the run passed the countable
    // limit and no identity over it is a proof.
    if (total_device == UINT32_MAX || dropped_device == UINT32_MAX || overflow_device == UINT32_MAX) {
        LOG_ERROR(
            "dep_gen reconcile: a device counter reached its counting limit (total=%lu dropped=%lu overflow=%lu) — "
            "this run's counts are unknown",
            static_cast<unsigned long>(total_device), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(overflow_device)
        );
        out.counters_unknown = true;
        clean = false;
    }

    if (dropped_device > 0) {
        out.device_dropped = true;
        LOG_WARN(
            "dep_gen reconcile: %lu records dropped on device side (free_queue empty or ready_queue full). "
            "Increase PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE / PLATFORM_DEP_GEN_READYQUEUE_SIZE if frequent. "
            "deps.json will NOT be emitted for this run (incomplete graph).",
            static_cast<unsigned long>(dropped_device)
        );
        clean = false;
    }
    // collected counts physical buffer slots; total_device counts submits; the
    // chain expands submits into multiple slots, so the overflow counter
    // bridges the two.
    if (total_collected_ + dropped_device != total_device + overflow_device) {
        out.identity_broken = true;
        LOG_WARN(
            "dep_gen reconcile: record count mismatch (collected=%lu + dropped=%lu != device_total=%lu + "
            "overflow=%lu, silent_loss=%ld)",
            static_cast<unsigned long>(total_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(total_device), static_cast<unsigned long>(overflow_device),
            static_cast<long>(total_device + overflow_device) - static_cast<long>(total_collected_ + dropped_device)
        );
        clean = false;
    } else {
        LOG_INFO(
            "dep_gen reconcile: counts match (collected=%lu, dropped=%lu, device_total=%lu, overflow=%lu)",
            static_cast<unsigned long>(total_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(total_device), static_cast<unsigned long>(overflow_device)
        );
    }

    out.clean = clean;
    return out;
}

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

void DepGenCollector::finalize(DepGenUnregisterCallback unregister_cb, const DepGenFreeCallback &free_cb) {
    if (!initialized_) return;

    stop();

    // `stop()` is the reader join, so this is the first point at which the
    // quarantined host copies are nobody's to append to. Disposing them here
    // returns their bytes and clears the admission block a quarantine holds;
    // it settles host storage only — the sticky diagnostic error survives, and
    // the pooled device buffers are released below by their own proof.
    discard_quarantined_runs();

    {
        std::scoped_lock lock(records_mutex_);
        records_by_run_.clear();
    }

    auto release_dev = [&](void *p) {
        release_one_buffer(p, unregister_cb, free_cb);
    };
    std::unordered_set<void *> released_dev_ptrs;
    auto release_dev_once = [&](void *p) {
        if (p == nullptr || !released_dev_ptrs.insert(p).second) return;
        release_dev(p);
    };

    // Free buffers still parked in the free_queue / current_buf_ptr.
    // Release the device pointer only — the paired host shadow stays in
    // dev_to_host_ and is freed by clear_mappings() below.
    if (shm_host_ != nullptr) {
        if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
            profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
        }
        rmb();
        DepGenBufferState *state = dep_gen_state(0);

        release_dev_once(reinterpret_cast<void *>(state->current_buf_ptr));
        state->current_buf_ptr = 0;

        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_DEP_GEN_SLOT_COUNT) queued = PLATFORM_DEP_GEN_SLOT_COUNT;
        for (uint32_t i = 0; i < queued; i++) {
            uint32_t slot = (head + i) % PLATFORM_DEP_GEN_SLOT_COUNT;
            release_dev_once(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
            state->free_queue.buffer_ptrs[slot] = 0;
        }
        state->free_queue.head = tail;
    }

    // Release framework-owned device allocations (recycled pool,
    // ready_queue, done_queue). Host shadows are freed by clear_mappings().
    manager_.release_owned_buffers([&](void *p) {
        release_dev_once(p);
    });

    // Free shared header region (device only — shadow stays in
    // dev_to_host_ until clear_mappings).
    if (shm_dev_ != nullptr) {
        release_dev(shm_dev_);
        shm_dev_ = nullptr;
    }

    // Free remaining host shadows (per-state buffers + shm region).
    manager_.clear_mappings();

    initialized_ = false;
    shm_size_ = 0;
    total_collected_ = 0;
    clear_memory_context();
    LOG_INFO("DepGen collector finalized");
}
