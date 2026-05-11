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
 */

#include "host/dep_gen_collector.h"

#include <cassert>
#include <cstring>
#include <unordered_set>

#include "common/memory_barrier.h"
#include "common/unified_log.h"

DepGenCollector::~DepGenCollector() { stop(); }

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

int DepGenCollector::init(
    int num_threads, DepGenAllocCallback alloc_cb, DepGenRegisterCallback register_cb, DepGenFreeCallback free_cb,
    void *user_data, int device_id
) {
    if (num_threads <= 0 || alloc_cb == nullptr || free_cb == nullptr) {
        LOG_ERROR("DepGenCollector::init: invalid arguments");
        return -1;
    }

    num_threads_ = num_threads;
    buffers_registered_ = (register_cb != nullptr);
    total_collected_ = 0;
    records_.clear();
    execution_complete_.store(false, std::memory_order_release);

    // ---- Allocate shared header + buffer-state region ----
    // dep_gen is single-instance: just one DepGenBufferState after the header.
    const int num_instances = 1;
    shm_size_ = calc_dep_gen_shm_size(num_instances);
    shm_dev_ = alloc_cb(shm_size_, user_data);
    if (shm_dev_ == nullptr) {
        LOG_ERROR("DepGenCollector: failed to allocate dep_gen shared memory (%zu bytes)", shm_size_);
        return -1;
    }

    if (register_cb != nullptr) {
        int rc = register_cb(shm_dev_, shm_size_, device_id, &shm_host_);
        if (rc != 0) {
            LOG_ERROR("DepGenCollector: halHostRegister for dep_gen SHM failed: %d", rc);
            free_cb(shm_dev_, user_data);
            shm_dev_ = nullptr;
            return rc;
        }
        shm_registered_ = true;
    } else {
        shm_host_ = shm_dev_;
    }
    std::memset(shm_host_, 0, shm_size_);

    DepGenDataHeader *hdr = get_dep_gen_header(shm_host_);
    hdr->num_instances = static_cast<uint32_t>(num_instances);

    // ---- Allocate DepGenBuffers, populate free_queue + recycled pool ----
    const size_t buf_size = sizeof(DepGenBuffer);
    DepGenBufferState *state = dep_gen_state(0);

    for (int b = 0; b < PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE; b++) {
        void *dev_ptr = alloc_cb(buf_size, user_data);
        if (dev_ptr == nullptr) {
            LOG_ERROR("DepGenCollector: failed to allocate DepGenBuffer b=%d", b);
            return -1;
        }

        void *host_ptr = dev_ptr;
        if (register_cb != nullptr) {
            int rc = register_cb(dev_ptr, buf_size, device_id, &host_ptr);
            if (rc != 0) {
                LOG_ERROR("DepGenCollector: halHostRegister for DepGenBuffer b=%d failed: %d", b, rc);
                free_cb(dev_ptr, user_data);
                return rc;
            }
        }
        std::memset(host_ptr, 0, buf_size);

        manager_.register_mapping(dev_ptr, host_ptr);

        if (b < PLATFORM_DEP_GEN_SLOT_COUNT) {
            uint32_t tail = state->free_queue.tail;
            assert(tail - state->free_queue.head < PLATFORM_DEP_GEN_SLOT_COUNT && "free_queue overflow on init");
            state->free_queue.buffer_ptrs[tail % PLATFORM_DEP_GEN_SLOT_COUNT] = reinterpret_cast<uint64_t>(dev_ptr);
            wmb();
            state->free_queue.tail = tail + 1;
            wmb();
        } else {
            manager_.push_recycled(0, dev_ptr);
        }
    }

    initialized_ = true;
    set_memory_context(alloc_cb, register_cb, free_cb, user_data, shm_host_, device_id);

    LOG_INFO_V0(
        "DepGen collector initialized: %d threads, SHM=0x%lx (records held in memory until replay)", num_threads,
        reinterpret_cast<unsigned long>(shm_dev_)
    );
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

    rmb();

    bool clean = true;

    DepGenBufferState *state = dep_gen_state(0);
    uint64_t buf_dev = state->current_buf_ptr;
    if (buf_dev != 0) {
        void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_dev));
        if (host_ptr != nullptr) {
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

    if (dropped_device > 0) {
        LOG_WARN(
            "dep_gen reconcile: %lu records dropped on device side (free_queue empty or ready_queue full). "
            "Increase PLATFORM_DEP_GEN_BUFFERS_PER_INSTANCE / PLATFORM_DEP_GEN_READYQUEUE_SIZE if frequent. "
            "deps.json will NOT be emitted for this run (incomplete graph).",
            static_cast<unsigned long>(dropped_device)
        );
        clean = false;
    }
    if (total_collected_ + dropped_device != total_device) {
        LOG_WARN(
            "dep_gen reconcile: record count mismatch (collected=%lu + dropped=%lu != device_total=%lu, "
            "silent_loss=%ld)",
            static_cast<unsigned long>(total_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(total_device),
            static_cast<long>(total_device) - static_cast<long>(total_collected_ + dropped_device)
        );
        clean = false;
    } else {
        LOG_INFO_V0(
            "dep_gen reconcile: counts match (collected=%lu, dropped=%lu, device_total=%lu)",
            static_cast<unsigned long>(total_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(total_device)
        );
    }

    return clean;
}

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

void DepGenCollector::finalize(DepGenUnregisterCallback unregister_cb, DepGenFreeCallback free_cb, void *user_data) {
    if (!initialized_) return;

    stop();

    {
        std::scoped_lock lock(records_mutex_);
        records_.clear();
        records_.shrink_to_fit();
    }

    // Same pattern as PmuCollector: walk owned buffers, then the free_queue
    // and current_buf_ptr, releasing each unique device pointer once.
    auto release_buf = [&](void *p) {
        if (buffers_registered_ && unregister_cb != nullptr) {
            unregister_cb(p, device_id_);
        }
        if (free_cb != nullptr) {
            free_cb(p, user_data);
        }
    };
    manager_.release_owned_buffers(release_buf);

    if (shm_host_ != nullptr) {
        std::unordered_set<void *> already_freed;
        auto release_unique = [&](void *p) {
            if (p == nullptr || !already_freed.insert(p).second) return;
            release_buf(p);
        };
        DepGenBufferState *state = dep_gen_state(0);
        release_unique(reinterpret_cast<void *>(state->current_buf_ptr));
        state->current_buf_ptr = 0;
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_DEP_GEN_SLOT_COUNT) queued = PLATFORM_DEP_GEN_SLOT_COUNT;
        for (uint32_t i = 0; i < queued; i++) {
            uint32_t slot = (head + i) % PLATFORM_DEP_GEN_SLOT_COUNT;
            release_unique(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
            state->free_queue.buffer_ptrs[slot] = 0;
        }
        state->free_queue.head = tail;
    }
    manager_.clear_mappings();

    if (shm_dev_ != nullptr) {
        if (shm_registered_ && unregister_cb != nullptr) {
            unregister_cb(shm_dev_, device_id_);
        }
        if (free_cb != nullptr) {
            free_cb(shm_dev_, user_data);
        }
        shm_dev_ = nullptr;
        shm_host_ = nullptr;
    }

    initialized_ = false;
    buffers_registered_ = false;
    shm_registered_ = false;
    shm_size_ = 0;
    total_collected_ = 0;
    clear_memory_context();
    LOG_INFO_V0("DepGen collector finalized");
}
