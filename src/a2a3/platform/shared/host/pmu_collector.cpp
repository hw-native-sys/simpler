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
 * @file pmu_collector.cpp
 * @brief Host-side PMU collector. The mgmt-thread + buffer-pool machinery
 *        lives in profiling_common::BufferPoolManager parameterized by
 *        PmuModule (host/pmu_collector.h); this file owns the per-buffer
 *        on_buffer_collected callback (CSV output) and the device-side
 *        cross-check. The poll loop itself lives in
 *        profiling_common::ProfilerBase.
 */

#include "host/pmu_collector.h"

#include <array>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>
#include <unordered_set>

#include "common/memory_barrier.h"
#include "common/unified_log.h"

namespace {

int owner_recycled_shard_for_core(int core_index, int thread_count) {
    int cluster_index = core_index / PLATFORM_CORES_PER_BLOCKDIM;
    return cluster_index % thread_count;
}

bool recycled_seed_capacity_is_sufficient(int num_cores, int thread_count, int surplus_per_core, size_t capacity) {
    if (surplus_per_core <= 0) return true;
    std::array<int, PLATFORM_MAX_AICPU_THREADS> per_shard{};
    for (int core = 0; core < num_cores; core++) {
        per_shard[static_cast<size_t>(owner_recycled_shard_for_core(core, thread_count))] += surplus_per_core;
    }

    bool ok = true;
    for (int shard = 0; shard < thread_count; shard++) {
        if (static_cast<size_t>(per_shard[static_cast<size_t>(shard)]) <= capacity) continue;
        LOG_ERROR(
            "PMU recycled seed exceeds lane capacity: shard=%d need=%d capacity=%zu "
            "(num_cores=%d thread_count=%d)",
            shard, per_shard[static_cast<size_t>(shard)], capacity, num_cores, thread_count
        );
        ok = false;
    }
    return ok;
}

}  // namespace

PmuCollector::~PmuCollector() { stop(); }

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

int PmuCollector::init(
    int num_cores, int num_threads, const std::string &csv_path, PmuEventType event_type,
    const PmuAllocCallback &alloc_cb, PmuRegisterCallback register_cb, const PmuFreeCallback &free_cb, int device_id
) {
    if (num_cores <= 0 || num_threads <= 0 || alloc_cb == nullptr || free_cb == nullptr) {
        LOG_ERROR("PmuCollector::init: invalid arguments");
        return -1;
    }
    if (num_cores > PLATFORM_MAX_CORES || num_threads > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR(
            "PmuCollector::init: dimensions out of range (cores=%d/%d threads=%d/%d)", num_cores, PLATFORM_MAX_CORES,
            num_threads, PLATFORM_MAX_AICPU_THREADS
        );
        return -1;
    }

    // Must precede the recycled-lane seeding below: push_recycled() folds its
    // shard argument modulo the manager's shard count.
    set_aicpu_thread_num(num_threads);

    num_cores_ = num_cores;
    num_threads_ = num_threads;
    event_type_ = event_type;
    csv_path_ = csv_path;
    buffers_registered_ = (register_cb != nullptr);

    reset_collector_shards();
    execution_complete_.store(false, std::memory_order_release);
    if (csv_file_.is_open()) {
        csv_file_.close();
    }
    constexpr int kPmuSurplusPerCore = (PLATFORM_PMU_BUFFERS_PER_CORE > PLATFORM_PMU_SLOT_COUNT) ?
                                           (PLATFORM_PMU_BUFFERS_PER_CORE - PLATFORM_PMU_SLOT_COUNT) :
                                           0;
    if (!recycled_seed_capacity_is_sufficient(
            num_cores, num_threads, kPmuSurplusPerCore, decltype(manager_)::kRecycledQueueCapacity
        )) {
        return -1;
    }

    // ---- Allocate shared header + buffer-state region ----
    shm_size_ = calc_pmu_data_size(num_cores);
    shm_dev_ = alloc_cb(shm_size_);
    if (shm_dev_ == nullptr) {
        LOG_ERROR("PmuCollector: failed to allocate PMU shared memory (%zu bytes)", shm_size_);
        return -1;
    }

    if (register_cb != nullptr) {
        int rc = register_cb(shm_dev_, shm_size_, device_id, &shm_host_);
        if (rc != 0) {
            LOG_ERROR("PmuCollector: halHostRegister for PMU SHM failed: %d", rc);
            free_cb(shm_dev_);
            shm_dev_ = nullptr;
            return rc;
        }
        shm_registered_ = true;
    } else {
        shm_host_ = shm_dev_;
    }
    std::memset(shm_host_, 0, shm_size_);

    PmuDataHeader *hdr = get_pmu_header(shm_host_);
    hdr->event_type = static_cast<uint32_t>(event_type);
    hdr->num_cores = static_cast<uint32_t>(num_cores);

    // ---- Allocate per-core PmuBuffers and populate free_queues + recycled pool ----
    const size_t buf_size = sizeof(PmuBuffer);

    for (int c = 0; c < num_cores; c++) {
        PmuBufferState *state = pmu_state(c);

        for (int b = 0; b < PLATFORM_PMU_BUFFERS_PER_CORE; b++) {
            void *dev_ptr = alloc_cb(buf_size);
            if (dev_ptr == nullptr) {
                LOG_ERROR("PmuCollector: failed to allocate PmuBuffer c=%d b=%d", c, b);
                return -1;
            }

            void *host_ptr = dev_ptr;
            if (register_cb != nullptr) {
                int rc = register_cb(dev_ptr, buf_size, device_id, &host_ptr);
                if (rc != 0) {
                    LOG_ERROR("PmuCollector: halHostRegister for PmuBuffer c=%d b=%d failed: %d", c, b, rc);
                    free_cb(dev_ptr);
                    return rc;
                }
            }
            std::memset(host_ptr, 0, buf_size);

            // Track dev→host so the mgmt thread can resolve_host_ptr().
            manager_.register_mapping(dev_ptr, host_ptr);

            if (b < PLATFORM_PMU_SLOT_COUNT) {
                // First N buffers go directly into this core's free_queue.
                uint32_t tail = state->free_queue.tail;
                assert(tail - state->free_queue.head < PLATFORM_PMU_SLOT_COUNT && "free_queue overflow on init");
                state->free_queue.buffer_ptrs[tail % PLATFORM_PMU_SLOT_COUNT] = reinterpret_cast<uint64_t>(dev_ptr);
                wmb();
                state->free_queue.tail = tail + 1;
                wmb();
            } else {
                // Surplus buffers go into the cluster owner shard so runtime
                // drain consumes the same recycled lane it owns.
                int shard = owner_recycled_shard_for_core(c, num_threads);
                if (!manager_.push_recycled(0, dev_ptr, shard)) {
                    (void)manager_.retire_unqueued_buffer(0, dev_ptr, shard);
                }
            }
        }
    }

    // ---- Build CSV header string ----
    {
        std::string header = "thread_id,core_id,task_id,func_id,core_type,pmu_total_cycles";
        const PmuEventConfig *evt = pmu_resolve_event_config_a2a3(event_type);
        if (evt == nullptr) {
            evt = &PMU_EVENTS_A2A3_PIPE_UTIL;
        }
        for (int i = 0; i < PMU_COUNTER_COUNT_A2A3; i++) {
            const char *name = evt->counter_names[i];
            if (name == nullptr || name[0] == '\0') {
                continue;
            }
            header += ',';
            header += name;
        }
        header += ",event_type\n";
        csv_header_ = std::move(header);
    }

    initialized_ = true;
    // Hand the memory context to the base. start(tf) (inherited) will assemble
    // a MemoryOps from these and launch mgmt + poll threads. PmuModule's alloc
    // fallback in process_entry can then grow the buffer pool on demand if
    // both the per-core free_queue and the recycled pool drain.
    set_memory_context(
        alloc_cb, register_cb, free_cb, /*copy_to=*/nullptr, /*copy_from=*/nullptr, shm_dev_, shm_host_, shm_size_,
        device_id
    );

    LOG_INFO_V0(
        "PMU collector initialized: %d cores, %d threads, SHM=0x%lx, CSV=%s (opened on first record)", num_cores,
        num_threads, reinterpret_cast<unsigned long>(shm_dev_), csv_path_.c_str()
    );
    return 0;
}

void PmuCollector::start(const profiling_common::ThreadFactory &thread_factory) {
    if (shm_host_ == nullptr) return;
    reset_collector_shards();
    profiling_common::ProfilerBase<PmuCollector, PmuModule>::start(thread_factory);
}

// ---------------------------------------------------------------------------
// CSV writing
// ---------------------------------------------------------------------------

size_t PmuCollector::normalize_collector_shard(int collector_shard) const {
    const size_t shard_count = csv_shard_files_.size();
    const bool valid_shard = collector_shard >= 0 && static_cast<size_t>(collector_shard) < shard_count;
    if (!valid_shard) {
        assert(false && "collector_shard out of range");
        return shard_count;
    }
    return static_cast<size_t>(collector_shard);
}

bool PmuCollector::close_csv_shards() {
    bool success = true;
    for (size_t shard = 0; shard < csv_shard_files_.size(); shard++) {
        auto &file = csv_shard_files_[shard];
        if (file.is_open()) {
            file.flush();
            if (!file.good()) {
                LOG_ERROR("PmuCollector: failed to flush CSV shard file: %s", csv_shard_paths_[shard].c_str());
                success = false;
            }
            file.close();
            if (file.fail()) {
                LOG_ERROR("PmuCollector: failed to close CSV shard file: %s", csv_shard_paths_[shard].c_str());
                success = false;
            }
        }
    }
    if (!success) {
        csv_shard_io_failed_.store(true, std::memory_order_relaxed);
    }
    return success;
}

void PmuCollector::cleanup_csv_shards() {
    for (const auto &path : csv_shard_paths_) {
        if (path.empty()) continue;
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
}

void PmuCollector::reset_collector_shards() {
    (void)close_csv_shards();
    cleanup_csv_shards();

    const size_t shard_count = static_cast<size_t>(manager_.shard_count());
    csv_shard_paths_.clear();
    csv_shard_paths_.reserve(shard_count);
    for (size_t shard = 0; shard < shard_count; shard++) {
        csv_shard_paths_.push_back(csv_path_ + ".shard" + std::to_string(shard) + ".tmp");
    }
    csv_shard_files_.clear();
    csv_shard_files_.resize(shard_count);
    collector_counters_.assign(shard_count, {});
    total_collected_ = 0;
    csv_shard_io_failed_.store(false, std::memory_order_relaxed);
    csv_shards_finalized_ = false;
}

bool PmuCollector::ensure_csv_open() {
    if (csv_file_.is_open()) return csv_file_.good();
    csv_file_.clear();
    csv_file_.open(csv_path_, std::ios::out | std::ios::trunc);
    if (!csv_file_.is_open()) {
        LOG_ERROR("PmuCollector: failed to open CSV file: %s", csv_path_.c_str());
        return false;
    }
    csv_file_ << csv_header_;
    if (!csv_file_.good()) {
        LOG_ERROR("PmuCollector: failed to write CSV header: %s", csv_path_.c_str());
        return false;
    }
    return true;
}

bool PmuCollector::ensure_csv_shard_open(size_t shard) {
    if (shard >= csv_shard_files_.size() || shard >= csv_shard_paths_.size()) {
        LOG_ERROR("PmuCollector: invalid CSV shard index %zu", shard);
        csv_shard_io_failed_.store(true, std::memory_order_relaxed);
        return false;
    }
    auto &file = csv_shard_files_[shard];
    if (file.is_open()) return true;
    file.open(csv_shard_paths_[shard], std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        LOG_ERROR("PmuCollector: failed to open CSV shard file: %s", csv_shard_paths_[shard].c_str());
        csv_shard_io_failed_.store(true, std::memory_order_relaxed);
        return false;
    }
    return true;
}

void PmuCollector::append_buffer_to_csv_shard(
    int core_id, int thread_idx, const void *buf_host_ptr, int collector_shard
) {
    const PmuBuffer *buf = reinterpret_cast<const PmuBuffer *>(buf_host_ptr);
    uint32_t n = buf->count;
    if (n > static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER)) {
        n = static_cast<uint32_t>(PLATFORM_PMU_RECORDS_PER_BUFFER);
    }
    if (n == 0) return;

    const size_t shard = normalize_collector_shard(collector_shard);
    if (!ensure_csv_shard_open(shard)) return;

    const PmuEventConfig *evt = pmu_resolve_event_config_a2a3(event_type_);
    if (evt == nullptr) {
        evt = &PMU_EVENTS_A2A3_PIPE_UTIL;
    }

    auto &rows = csv_shard_files_[shard];
    for (uint32_t i = 0; i < n; i++) {
        const PmuRecord &r = buf->records[i];
        rows << thread_idx << ',' << core_id << ',';
        rows << "0x" << std::hex << std::setw(16) << std::setfill('0') << r.task_id << std::dec << std::setfill(' ');
        rows << ',' << r.func_id << ',' << static_cast<int>(r.core_type) << ',' << r.pmu_total_cycles;
        for (int k = 0; k < PMU_COUNTER_COUNT_A2A3; k++) {
            const char *name = evt->counter_names[k];
            if (name == nullptr || name[0] == '\0') {
                continue;
            }
            rows << ',' << r.pmu_counters[k];
        }
        rows << ',' << static_cast<uint32_t>(event_type_) << '\n';
    }
    if (!rows.good()) {
        LOG_ERROR("PmuCollector: failed to write CSV shard file: %s", csv_shard_paths_[shard].c_str());
        csv_shard_io_failed_.store(true, std::memory_order_relaxed);
        return;
    }
    collector_counters_[shard].total_collected += n;
}

bool PmuCollector::flush_collector_shards_to_csv() {
    if (csv_shards_finalized_) {
        return true;
    }

    const bool shards_closed = close_csv_shards();
    if (!shards_closed || csv_shard_io_failed_.load(std::memory_order_relaxed)) {
        LOG_ERROR("PmuCollector: CSV shard output failed; preserving shard files for diagnosis");
        return false;
    }

    uint64_t collected_candidate = 0;
    bool has_rows = false;
    for (size_t shard = 0; shard < collector_counters_.size(); shard++) {
        collected_candidate += collector_counters_[shard].total_collected;
        has_rows = has_rows || collector_counters_[shard].total_collected != 0;
    }

    if (has_rows) {
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
        if (!ensure_csv_open()) {
            if (csv_file_.is_open()) {
                csv_file_.close();
                std::error_code ec;
                std::filesystem::remove(csv_path_, ec);
            }
            return false;
        }
        for (size_t shard = 0; shard < csv_shard_paths_.size(); shard++) {
            if (collector_counters_[shard].total_collected == 0) continue;
            std::ifstream shard_file(csv_shard_paths_[shard], std::ios::binary);
            if (!shard_file.is_open()) {
                LOG_ERROR("PmuCollector: failed to read CSV shard file: %s", csv_shard_paths_[shard].c_str());
                csv_file_.close();
                std::error_code ec;
                std::filesystem::remove(csv_path_, ec);
                return false;
            }
            csv_file_ << shard_file.rdbuf();
            if (shard_file.bad() || !csv_file_.good()) {
                LOG_ERROR("PmuCollector: failed to merge CSV shard file: %s", csv_shard_paths_[shard].c_str());
                csv_file_.close();
                std::error_code ec;
                std::filesystem::remove(csv_path_, ec);
                return false;
            }
        }
        csv_file_.flush();
        if (!csv_file_.good()) {
            LOG_ERROR("PmuCollector: failed to flush merged CSV file: %s", csv_path_.c_str());
            csv_file_.close();
            std::error_code ec;
            std::filesystem::remove(csv_path_, ec);
            return false;
        }
    }
    total_collected_ = collected_candidate;
    cleanup_csv_shards();
    csv_shards_finalized_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// ProfilerBase callback
// ---------------------------------------------------------------------------

void PmuCollector::on_buffer_collected(const PmuReadyBufferInfo &info, int collector_shard) {
    append_buffer_to_csv_shard(
        static_cast<int>(info.core_index), static_cast<int>(info.thread_index), info.host_buffer_ptr, collector_shard
    );
}

// ---------------------------------------------------------------------------
// reconcile_counters: passive sanity-check + device-side cross-check
// ---------------------------------------------------------------------------
//
// Host never recovers records from device-side current_buf_ptr. Device flush
// (pmu_aicpu_flush_buffers) is the only data path: a flush failure must bump
// dropped_record_count and clear current_buf_ptr on the device side. Host's
// job here is purely accounting + sanity assertion — recovering would mask
// AICPU flush bugs.

void PmuCollector::reconcile_counters() {
    if (shm_host_ == nullptr) return;

    rmb();
    flush_collector_shards_to_csv();

    // After stop(), pmu_aicpu_flush_buffers should have either enqueued the
    // active buffer (success → current_buf_ptr=0) or counted it as dropped
    // and cleared it. A non-zero pointer with non-zero count means records
    // AICPU neither delivered nor accounted for — a device-side flush bug.
    for (int c = 0; c < num_cores_; c++) {
        PmuBufferState *state = pmu_state(c);
        uint64_t buf_dev = state->current_buf_ptr;
        if (buf_dev == 0) continue;

        void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_dev));
        if (host_ptr == nullptr) continue;

        uint32_t count = reinterpret_cast<const PmuBuffer *>(host_ptr)->count;
        if (count == 0) continue;

        LOG_ERROR(
            "PMU reconcile: core %d has un-flushed buffer (current_buf_ptr=0x%lx, count=%u) after "
            "stop() — device flush failed",
            c, static_cast<unsigned long>(buf_dev), count
        );
    }

    // Cross-check device-side totals against what we wrote to CSV.
    uint64_t total_device = 0;
    uint64_t dropped_device = 0;
    for (int c = 0; c < num_cores_; c++) {
        PmuBufferState *state = pmu_state(c);
        total_device += state->total_record_count;
        dropped_device += state->dropped_record_count;
    }

    if (dropped_device > 0) {
        LOG_WARN(
            "PMU reconcile: %lu records dropped on device side (free_queue empty or ready_queue full). "
            "Increase PLATFORM_PMU_BUFFERS_PER_CORE / PLATFORM_PMU_READYQUEUE_SIZE if this is frequent.",
            static_cast<unsigned long>(dropped_device)
        );
    }
    if (total_collected_ + dropped_device != total_device) {
        LOG_WARN(
            "PMU reconcile: record count mismatch (collected=%lu + dropped=%lu != device_total=%lu, "
            "silent_loss=%ld) — AICore/AICPU race",
            static_cast<unsigned long>(total_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(total_device),
            static_cast<long>(total_device) - static_cast<long>(total_collected_ + dropped_device)
        );
    } else {
        LOG_INFO_V0(
            "PMU reconcile: record counts match (collected=%lu, dropped=%lu, device_total=%lu)",
            static_cast<unsigned long>(total_collected_), static_cast<unsigned long>(dropped_device),
            static_cast<unsigned long>(total_device)
        );
    }
}

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

void PmuCollector::finalize(PmuUnregisterCallback unregister_cb, const PmuFreeCallback &free_cb) {
    if (!initialized_) return;

    // Stop mgmt + collector threads if the caller didn't already (idempotent).
    stop();
    flush_collector_shards_to_csv();

    if (csv_file_.is_open()) {
        csv_file_.close();
    }

    // Release per-buffer mappings tracked by the framework. PMU registered
    // each PmuBuffer individually at init time (when register_cb was set), so
    // unregister each one here before freeing.
    auto release_buf = [&](void *p) {
        release_one_buffer(p, buffers_registered_ ? unregister_cb : nullptr, free_cb);
    };
    manager_.release_owned_buffers(release_buf);

    // Buffers still parked in per-core free_queues / current_buf_ptr.
    if (shm_host_ != nullptr) {
        std::unordered_set<void *> already_freed;
        auto release_unique = [&](void *p) {
            if (p == nullptr || !already_freed.insert(p).second) return;
            release_buf(p);
        };
        for (int c = 0; c < num_cores_; c++) {
            PmuBufferState *state = pmu_state(c);
            release_unique(reinterpret_cast<void *>(state->current_buf_ptr));
            state->current_buf_ptr = 0;
            rmb();
            uint32_t head = state->free_queue.head;
            uint32_t tail = state->free_queue.tail;
            uint32_t queued = tail - head;
            if (queued > PLATFORM_PMU_SLOT_COUNT) queued = PLATFORM_PMU_SLOT_COUNT;
            for (uint32_t i = 0; i < queued; i++) {
                uint32_t slot = (head + i) % PLATFORM_PMU_SLOT_COUNT;
                release_unique(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
                state->free_queue.buffer_ptrs[slot] = 0;
            }
            state->free_queue.head = tail;
        }
    }
    manager_.clear_mappings();

    // Free shared header region via the shared RAII helper.
    if (shm_dev_ != nullptr) {
        release_one_buffer(shm_dev_, shm_registered_ ? unregister_cb : nullptr, free_cb);
        shm_dev_ = nullptr;
        shm_host_ = nullptr;
    }

    initialized_ = false;
    (void)close_csv_shards();
    if (csv_shards_finalized_) {
        cleanup_csv_shards();
    }
    csv_shard_paths_.clear();
    csv_shard_paths_.shrink_to_fit();
    csv_shard_files_.clear();
    csv_shard_files_.shrink_to_fit();
    collector_counters_.clear();
    collector_counters_.shrink_to_fit();
    csv_shards_finalized_ = false;
    clear_memory_context();
}
