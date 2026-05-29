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
 * @file scope_stats_collector.cpp
 * @brief Host-side scope_stats collector. The mgmt-thread + buffer-pool
 *        machinery lives in profiling_common::BufferPoolManager parameterized
 *        by ScopeStatsModule (host/scope_stats_collector.h); this file owns the
 *        per-buffer on_buffer_collected callback (in-memory append), the
 *        device-side cross-check, and the NDJSON export.
 */

#include "host/scope_stats_collector.h"

#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <unordered_set>

#include "common/memory_barrier.h"
#include "common/unified_log.h"

ScopeStatsCollector::~ScopeStatsCollector() { stop(); }

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

int ScopeStatsCollector::init(
    int num_threads, const ScopeStatsAllocCallback &alloc_cb, ScopeStatsRegisterCallback register_cb,
    const ScopeStatsFreeCallback &free_cb, int device_id
) {
    if (num_threads <= 0 || alloc_cb == nullptr || free_cb == nullptr) {
        LOG_ERROR("ScopeStatsCollector::init: invalid arguments");
        return -1;
    }

    num_threads_ = num_threads;
    buffers_registered_ = (register_cb != nullptr);
    total_collected_ = 0;
    records_.clear();
    execution_complete_.store(false, std::memory_order_release);

    const int num_instances = 1;
    shm_size_ = calc_scope_stats_shm_size(num_instances);
    shm_dev_ = alloc_cb(shm_size_);
    if (shm_dev_ == nullptr) {
        LOG_ERROR("ScopeStatsCollector: failed to allocate scope_stats shared memory (%zu bytes)", shm_size_);
        return -1;
    }

    if (register_cb != nullptr) {
        int rc = register_cb(shm_dev_, shm_size_, device_id, &shm_host_);
        if (rc != 0) {
            LOG_ERROR("ScopeStatsCollector: halHostRegister for scope_stats SHM failed: %d", rc);
            free_cb(shm_dev_);
            shm_dev_ = nullptr;
            return rc;
        }
        shm_registered_ = true;
    } else {
        shm_host_ = shm_dev_;
    }
    std::memset(shm_host_, 0, shm_size_);

    ScopeStatsDataHeader *hdr = get_scope_stats_header(shm_host_);
    hdr->num_instances = static_cast<uint32_t>(num_instances);

    const size_t buf_size = sizeof(ScopeStatsBuffer);
    ScopeStatsBufferState *state = scope_stats_state(0);

    for (int b = 0; b < PLATFORM_SCOPE_STATS_BUFFERS_PER_INSTANCE; b++) {
        void *dev_ptr = alloc_cb(buf_size);
        if (dev_ptr == nullptr) {
            LOG_ERROR("ScopeStatsCollector: failed to allocate ScopeStatsBuffer b=%d", b);
            return -1;
        }

        void *host_ptr = dev_ptr;
        if (register_cb != nullptr) {
            int rc = register_cb(dev_ptr, buf_size, device_id, &host_ptr);
            if (rc != 0) {
                LOG_ERROR("ScopeStatsCollector: halHostRegister for ScopeStatsBuffer b=%d failed: %d", b, rc);
                free_cb(dev_ptr);
                return rc;
            }
        }
        std::memset(host_ptr, 0, buf_size);

        manager_.register_mapping(dev_ptr, host_ptr);

        if (b < PLATFORM_SCOPE_STATS_SLOT_COUNT) {
            uint32_t tail = state->free_queue.tail;
            assert(tail - state->free_queue.head < PLATFORM_SCOPE_STATS_SLOT_COUNT && "free_queue overflow on init");
            state->free_queue.buffer_ptrs[tail % PLATFORM_SCOPE_STATS_SLOT_COUNT] = reinterpret_cast<uint64_t>(dev_ptr);
            wmb();
            state->free_queue.tail = tail + 1;
            wmb();
        } else {
            manager_.push_recycled(0, dev_ptr);
        }
    }

    initialized_ = true;
    set_memory_context(alloc_cb, register_cb, free_cb, shm_host_, device_id);

    LOG_INFO_V0(
        "ScopeStats collector initialized: %d threads, SHM=0x%lx", num_threads,
        reinterpret_cast<unsigned long>(shm_dev_)
    );
    return 0;
}

// ---------------------------------------------------------------------------
// Record accumulation (in-memory)
// ---------------------------------------------------------------------------

void ScopeStatsCollector::append_buffer_records(const void *buf_host_ptr) {
    const ScopeStatsBuffer *buf = reinterpret_cast<const ScopeStatsBuffer *>(buf_host_ptr);
    uint32_t n = buf->count;
    if (n > static_cast<uint32_t>(PLATFORM_SCOPE_STATS_RECORDS_PER_BUFFER)) {
        n = static_cast<uint32_t>(PLATFORM_SCOPE_STATS_RECORDS_PER_BUFFER);
    }
    if (n == 0) return;

    std::scoped_lock lock(records_mutex_);
    records_.insert(records_.end(), buf->records, buf->records + n);
    total_collected_ += n;
}

void ScopeStatsCollector::on_buffer_collected(const ScopeStatsReadyBufferInfo &info) {
    append_buffer_records(info.host_buffer_ptr);
}

// ---------------------------------------------------------------------------
// reconcile_counters
// ---------------------------------------------------------------------------

bool ScopeStatsCollector::reconcile_counters() {
    if (shm_host_ == nullptr) return false;

    rmb();
    bool clean = true;

    ScopeStatsBufferState *state = scope_stats_state(0);
    uint64_t buf_dev = state->current_buf_ptr;
    if (buf_dev != 0) {
        void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_dev));
        if (host_ptr != nullptr) {
            uint32_t count = reinterpret_cast<const ScopeStatsBuffer *>(host_ptr)->count;
            if (count != 0) {
                LOG_ERROR(
                    "scope_stats reconcile: un-flushed buffer (current_buf_ptr=0x%lx, count=%u) — device flush failed",
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
            "scope_stats reconcile: %lu records dropped on device side (free_queue empty or ready_queue full). "
            "Increase PLATFORM_SCOPE_STATS_BUFFERS_PER_INSTANCE / PLATFORM_SCOPE_STATS_READYQUEUE_SIZE if frequent.",
            static_cast<unsigned long>(dropped_device)
        );
        clean = false;
    }
    if (total_collected_ + dropped_device != total_device) {
        LOG_WARN(
            "scope_stats reconcile: record count mismatch (collected=%lu + dropped=%lu != device_total=%lu)",
            static_cast<unsigned long>(total_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(total_device)
        );
        clean = false;
    } else {
        LOG_INFO_V0(
            "scope_stats reconcile: counts match (collected=%lu, dropped=%lu, device_total=%lu)",
            static_cast<unsigned long>(total_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(total_device)
        );
    }

    return clean;
}

// ---------------------------------------------------------------------------
// NDJSON export
// ---------------------------------------------------------------------------

int ScopeStatsCollector::write_jsonl(const std::string &output_dir) {
    if (!initialized_ || shm_host_ == nullptr) return 0;

    std::filesystem::path dir = std::filesystem::path(output_dir) / "scope_stats";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LOG_WARN("scope_stats: failed to create output dir %s: %s", dir.c_str(), ec.message().c_str());
    }
    const std::string path = (dir / "scope_stats.jsonl").string();

    std::FILE *fp = std::fopen(path.c_str(), "w");
    if (fp == nullptr) {
        LOG_ERROR("scope_stats: failed to open %s", path.c_str());
        return -1;
    }

    const ScopeStatsDataHeader *hdr = scope_stats_header();
    const ScopeStatsBufferState *state = scope_stats_state(0);

    // Line 1: run metadata. The per-ring maxima (task window / heap capacities)
    // and the tensormap capacity are run-constants, so they live here once
    // rather than on every record.
    std::string task_window_max;
    std::string heap_max;
    for (int r = 0; r < PTO2_SCOPE_STATS_MAX_RING_DEPTH; r++) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%s%d", r == 0 ? "" : ", ", hdr->task_window_cap[r]);
        task_window_max += buf;
        std::snprintf(buf, sizeof(buf), "%s%" PRIu64, r == 0 ? "" : ", ", hdr->heap_cap[r]);
        heap_max += buf;
    }
    std::fprintf(
        fp,
        "{\"version\": 4, \"fatal\": %s, \"dropped\": %u, \"total\": %u, "
        "\"task_window_max\": [%s], \"heap_max\": [%s], \"tensormap_max\": %d}\n",
        hdr->fatal_latched ? "true" : "false", state->dropped_record_count, state->total_record_count,
        task_window_max.c_str(), heap_max.c_str(), hdr->tensormap_cap
    );

    // Serialize every record into one in-memory buffer, then a single fwrite.
    // The hot loop is one snprintf per record (not 6 fprintf): stdio format
    // parsing + per-call FILE locking on ~6×N calls was the dominant host cost.
    std::scoped_lock lock(records_mutex_);
    std::string out;
    out.reserve(records_.size() * 128);
    char line[320];
    for (const ScopeStatsRecord &rec : records_) {
        const int site_len = static_cast<int>(strnlen(rec.site_file_basename, sizeof(rec.site_file_basename)));
        const char *phase = (rec.phase == SCOPE_STATS_PHASE_BEGIN) ? "begin" : "end";
        int n = std::snprintf(
            line, sizeof(line),
            "{\"site\": \"%.*s:%d\", \"phase\": \"%s\", \"depth\": %d, \"ring\": %d, "
            "\"task_window_start\": %d, \"task_window_end\": %d, "
            "\"heap_start\": %" PRIu64 ", \"heap_end\": %" PRIu64 ", "
            "\"tensormap\": %d}\n",
            site_len, rec.site_file_basename, rec.site_line, phase, rec.depth, rec.ring_id, rec.task_start,
            rec.task_end, rec.heap_start, rec.heap_end, rec.tensormap_used
        );
        if (n > 0) out.append(line, static_cast<size_t>(n < static_cast<int>(sizeof(line)) ? n : sizeof(line) - 1));
    }
    std::fwrite(out.data(), 1, out.size(), fp);
    std::fclose(fp);

    LOG_INFO_V1(
        "scope_stats: wrote %lu records (dropped=%u) to %s", static_cast<unsigned long>(records_.size()),
        state->dropped_record_count, path.c_str()
    );
    return 0;
}

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

void ScopeStatsCollector::finalize(ScopeStatsUnregisterCallback unregister_cb, const ScopeStatsFreeCallback &free_cb) {
    if (!initialized_) return;

    stop();

    {
        std::scoped_lock lock(records_mutex_);
        records_.clear();
        records_.shrink_to_fit();
    }

    auto release_buf = [&](void *p) {
        release_one_buffer(p, buffers_registered_ ? unregister_cb : nullptr, free_cb);
    };
    manager_.release_owned_buffers(release_buf);

    if (shm_host_ != nullptr) {
        std::unordered_set<void *> already_freed;
        auto release_unique = [&](void *p) {
            if (p == nullptr || !already_freed.insert(p).second) return;
            release_buf(p);
        };
        ScopeStatsBufferState *state = scope_stats_state(0);
        release_unique(reinterpret_cast<void *>(state->current_buf_ptr));
        state->current_buf_ptr = 0;
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_SCOPE_STATS_SLOT_COUNT) queued = PLATFORM_SCOPE_STATS_SLOT_COUNT;
        for (uint32_t i = 0; i < queued; i++) {
            uint32_t slot = (head + i) % PLATFORM_SCOPE_STATS_SLOT_COUNT;
            release_unique(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
            state->free_queue.buffer_ptrs[slot] = 0;
        }
        state->free_queue.head = tail;
    }
    manager_.clear_mappings();

    if (shm_dev_ != nullptr) {
        release_one_buffer(shm_dev_, shm_registered_ ? unregister_cb : nullptr, free_cb);
        shm_dev_ = nullptr;
        shm_host_ = nullptr;
    }

    initialized_ = false;
    buffers_registered_ = false;
    shm_registered_ = false;
    shm_size_ = 0;
    total_collected_ = 0;
    clear_memory_context();
    LOG_INFO_V0("ScopeStats collector finalized");
}
