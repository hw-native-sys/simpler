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
 *        device-side cross-check. Records stay in ``records_`` and are
 *        consumed directly by the host replay — no on-disk submit_trace.bin
 *        intermediary.
 *
 * a5 specifics: device↔host transfers go through profiling_copy.h. Each
 * DepGenBuffer's contents are pulled from device on demand inside
 * ProfilerAlgorithms::process_entry, so on_buffer_collected can read
 * `count` and `records[]` directly off the host shadow.
 */

#include "host/dep_gen_collector.h"

#include <cassert>
#include <cstring>

#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host/profiling_copy.h"

DepGenCollector::~DepGenCollector() { stop(); }

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

int DepGenCollector::init(
    int num_threads, const DepGenAllocCallback &alloc_cb, DepGenRegisterCallback register_cb,
    const DepGenFreeCallback &free_cb, int device_id
) {
    if (initialized_) {
        LOG_ERROR("DepGenCollector already initialized");
        return -1;
    }
    if (num_threads <= 0 || alloc_cb == nullptr || free_cb == nullptr) {
        LOG_ERROR("DepGenCollector::init: invalid arguments");
        return -1;
    }

    num_threads_ = num_threads;
    total_collected_ = 0;
    records_.clear();

    // Stash callbacks on the base up-front so alloc_paired_buffer sees
    // consistent values during init. shm_host_ stays nullptr until the shm
    // allocation succeeds — start(tf) gates on shm_host_.
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_for_ops, profiling_copy_from_device_for_ops,
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
        return -1;
    }

    std::memset(shm_host_local, 0, shm_size);
    DepGenDataHeader *hdr = get_dep_gen_header(shm_host_local);
    hdr->num_instances = static_cast<uint32_t>(num_instances);

    // ---- Allocate DepGenBuffers, populate free_queue + recycled pool ----
    const size_t buf_size = sizeof(DepGenBuffer);
    DepGenBufferState *state = get_dep_gen_buffer_state(shm_host_local, 0);

    for (int b = 0; b < PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE; b++) {
        void *host_ptr = nullptr;
        void *dev_ptr = alloc_paired_buffer(buf_size, &host_ptr);
        if (dev_ptr == nullptr) {
            LOG_ERROR("DepGenCollector: failed to allocate DepGenBuffer b=%d", b);
            return -1;
        }

        if (b < PLATFORM_DEP_GEN_SLOT_COUNT) {
            // First N buffers go directly into the (single) instance's free_queue.
            // All writes hit the host shadow here; the whole shm region gets
            // pushed to device once below — no wmb needed.
            uint32_t tail = state->free_queue.tail;
            assert(tail - state->free_queue.head < PLATFORM_DEP_GEN_SLOT_COUNT && "free_queue overflow on init");
            state->free_queue.buffer_ptrs[tail % PLATFORM_DEP_GEN_SLOT_COUNT] = reinterpret_cast<uint64_t>(dev_ptr);
            state->free_queue.tail = tail + 1;
        } else {
            manager_.push_recycled(0, dev_ptr);
        }
    }

    // Push the entire initialized shm region (header + BufferState +
    // free_queue contents) to device.
    profiling_copy_to_device(shm_dev_local, shm_host_local, shm_size);

    LOG_INFO_V0(
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
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_for_ops, profiling_copy_from_device_for_ops,
        shm_dev_local, shm_host_local, shm_size, device_id
    );
    initialized_ = true;
    return 0;
}

// ---------------------------------------------------------------------------
// Record accumulation (in-memory — no disk hop)
// ---------------------------------------------------------------------------

void DepGenCollector::append_buffer_records(const void *buf_host_ptr) {
    const DepGenBuffer *buf = reinterpret_cast<const DepGenBuffer *>(buf_host_ptr);
    uint32_t n = buf->count;
    if (n > static_cast<uint32_t>(PLATFORM_DEP_GEN_RECORDS_PER_BUFFER)) {
        n = static_cast<uint32_t>(PLATFORM_DEP_GEN_RECORDS_PER_BUFFER);
    }
    if (n == 0) return;

    std::scoped_lock lock(records_mutex_);
    records_.insert(records_.end(), buf->records, buf->records + n);
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

bool DepGenCollector::reconcile_counters() {
    if (shm_host_ == nullptr) return false;

    // mgmt thread is stopped by the caller; pull the latest BufferState
    // (current_buf_ptr, total/dropped counters) from device so the
    // cross-check sees post-stop() values.
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
    }

    bool clean = true;

    DepGenBufferState *state = dep_gen_state(0);
    uint64_t buf_dev = state->current_buf_ptr;
    if (buf_dev != 0) {
        void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_dev));
        if (host_ptr != nullptr) {
            profiling_copy_from_device(host_ptr, reinterpret_cast<void *>(buf_dev), sizeof(DepGenBuffer));
            uint32_t count = reinterpret_cast<const DepGenBuffer *>(host_ptr)->count;
            if (count != 0) {
                LOG_ERROR(
                    "dep_gen reconcile: un-flushed buffer (current_buf_ptr=0x%lx, count=%u) — device flush failed",
                    static_cast<unsigned long>(buf_dev), count
                );
                clean = false;
            }
        }
    }

    uint64_t total_device = state->total_record_count;
    uint64_t dropped_device = state->dropped_record_count;
    uint64_t overflow_device = state->total_overflow_record_count;

    if (dropped_device > 0) {
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
        LOG_WARN(
            "dep_gen reconcile: record count mismatch (collected=%lu + dropped=%lu != device_total=%lu + "
            "overflow=%lu, silent_loss=%ld)",
            static_cast<unsigned long>(total_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(total_device), static_cast<unsigned long>(overflow_device),
            static_cast<long>(total_device + overflow_device) - static_cast<long>(total_collected_ + dropped_device)
        );
        clean = false;
    } else {
        LOG_INFO_V0(
            "dep_gen reconcile: counts match (collected=%lu, dropped=%lu, device_total=%lu, overflow=%lu)",
            static_cast<unsigned long>(total_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(total_device), static_cast<unsigned long>(overflow_device)
        );
    }

    return clean;
}

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

void DepGenCollector::finalize(DepGenUnregisterCallback unregister_cb, const DepGenFreeCallback &free_cb) {
    if (!initialized_) return;

    stop();

    {
        std::scoped_lock lock(records_mutex_);
        records_.clear();
        records_.shrink_to_fit();
    }

    auto release_dev = [&](void *p) {
        release_one_buffer(p, unregister_cb, free_cb);
    };

    // Free buffers still parked in the free_queue / current_buf_ptr.
    // Release the device pointer only — the paired host shadow stays in
    // dev_to_host_ and is freed by clear_mappings() below.
    if (shm_host_ != nullptr) {
        DepGenBufferState *state = dep_gen_state(0);

        release_dev(reinterpret_cast<void *>(state->current_buf_ptr));
        state->current_buf_ptr = 0;

        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_DEP_GEN_SLOT_COUNT) queued = PLATFORM_DEP_GEN_SLOT_COUNT;
        for (uint32_t i = 0; i < queued; i++) {
            uint32_t slot = (head + i) % PLATFORM_DEP_GEN_SLOT_COUNT;
            release_dev(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
            state->free_queue.buffer_ptrs[slot] = 0;
        }
        state->free_queue.head = tail;
    }

    // Release framework-owned buffers (recycled pool, ready_queue,
    // done_queue). release_owned_buffers also frees their host shadows.
    manager_.release_owned_buffers([&](void *p) {
        release_dev(p);
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
    LOG_INFO_V0("DepGen collector finalized");
}
