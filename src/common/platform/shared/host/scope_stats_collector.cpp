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
 *
 * Memory mirroring is handled by the framework via the MemoryOps installed
 * at set_memory_context time:
 *   - SVM platforms (a2a3): copy_* not installed; profiling_copy_*_for_ops
 *     calls below reach the per-arch stubs that return 0; the host pointer
 *     IS the device pointer.
 *   - Non-SVM platforms (a5): copy_* installed; ProfilerAlgorithms pulls each
 *     ScopeStatsBuffer's contents from device on demand inside process_entry,
 *     so on_buffer_collected can read `count` and `records[]` directly off
 *     the host shadow.
 */

#include "host/scope_stats_collector.h"

#include <cassert>
#include <cinttypes>
#include <new>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>

#include "common/memory_barrier.h"
#include "common/unified_log.h"
#include "host/profiling_copy.h"
#include "../../../worker/runtime_c_api.h"

ScopeStatsCollector::~ScopeStatsCollector() {
    // The writer outlives the collector threads, so it is joined here too: a
    // joinable thread left behind at destruction terminates the process.
    stop_writer();
    stop();
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

int ScopeStatsCollector::init(
    int num_threads, const ScopeStatsAllocCallback &alloc_cb, ScopeStatsRegisterCallback register_cb,
    const ScopeStatsFreeCallback &free_cb, int device_id
) {
    if (num_threads <= 0 || num_threads > PLATFORM_MAX_AICPU_THREADS || alloc_cb == nullptr || free_cb == nullptr) {
        LOG_ERROR(
            "ScopeStatsCollector::init: invalid arguments (num_threads=%d, valid range: 1-%d)", num_threads,
            PLATFORM_MAX_AICPU_THREADS
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (initialized_) {
        // Already holding this run's device resources. They are not per-run:
        // configuration arrives via begin_run() and the layout is fixed at
        // compile time, so there is nothing here left to re-apply.
        return 0;
    }

    // Must precede the recycled-lane seeding below: push_recycled() folds its
    // shard argument modulo the manager's shard count.
    set_aicpu_thread_num(num_threads);

    total_collected_ = 0;
    refused_records_ = 0;
    (void)records_.release();
    recovered_current_buf_ = 0;
    recovered_current_total_ = 0;
    execution_complete_.store(false, std::memory_order_release);

    // Stash callbacks on the base up-front so alloc_paired_buffer sees
    // consistent values during init. shm_host_ stays nullptr until the shm
    // allocation succeeds — start(tf) gates on shm_host_.
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_or_null(), profiling_copy_from_device_or_null(),
        /*shm_dev=*/nullptr, /*shm_host=*/nullptr, /*shm_size=*/0, device_id
    );

    // RAII rollback: any early return after this point releases every
    // framework-tracked buffer (shm region + per-buffer-state PmuBuffer-style
    // entries) via free_cb. `guard.commit()` runs on the success path before
    // the trailing return 0.
    profiling_common::InitRollbackGuard<decltype(manager_)> guard(manager_, free_cb);

    const int num_instances = 1;
    size_t shm_size = calc_scope_stats_shm_size(num_instances);
    void *shm_host_local = nullptr;
    void *shm_dev_local = alloc_paired_buffer(shm_size, &shm_host_local);
    if (shm_dev_local == nullptr) {
        LOG_ERROR("ScopeStatsCollector: failed to allocate scope_stats shared memory (%zu bytes)", shm_size);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    std::memset(shm_host_local, 0, shm_size);
    ScopeStatsDataHeader *hdr = get_scope_stats_header(shm_host_local);
    hdr->num_instances = static_cast<uint32_t>(num_instances);

    const size_t buf_size = sizeof(ScopeStatsBuffer);
    ScopeStatsBufferState *state = get_scope_stats_buffer_state(shm_host_local, 0);

    const int owner_shard = (num_threads > 0) ? (num_threads - 1) : 0;
    for (int b = 0; b < PLATFORM_SCOPE_STATS_BUFFERS_PER_INSTANCE; b++) {
        void *host_ptr = nullptr;
        void *dev_ptr = alloc_paired_buffer(buf_size, &host_ptr);
        if (dev_ptr == nullptr) {
            LOG_ERROR("ScopeStatsCollector: failed to allocate ScopeStatsBuffer b=%d", b);
            return PTO_RUNTIME_ERR_INTERNAL;
        }

        if (b < PLATFORM_SCOPE_STATS_SLOT_COUNT) {
            uint32_t tail = state->free_queue.tail;
            assert(tail - state->free_queue.head < PLATFORM_SCOPE_STATS_SLOT_COUNT && "free_queue overflow on init");
            state->free_queue.buffer_ptrs[tail % PLATFORM_SCOPE_STATS_SLOT_COUNT] = reinterpret_cast<uint64_t>(dev_ptr);
            state->free_queue.tail = tail + 1;
        } else {
            if (!manager_.push_recycled(0, dev_ptr, owner_shard)) {
                (void)manager_.retire_unqueued_buffer(0, dev_ptr, owner_shard);
            }
        }
    }

    // Push the entire initialized shm region (header + BufferState +
    // free_queue contents) to device.
    profiling_copy_to_device(shm_dev_local, shm_host_local, shm_size);

    initialized_ = true;
    shm_dev_ = shm_dev_local;
    guard.commit();

    // Re-set_memory_context now that the shm region is ready. start(tf) gates
    // on shm_host_ being non-null, so this is the moment the collector becomes
    // startable.
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_or_null(), profiling_copy_from_device_or_null(),
        shm_dev_local, shm_host_local, shm_size, device_id
    );

    LOG_INFO(
        "ScopeStats collector initialized: %d threads, SHM=0x%lx", num_threads,
        reinterpret_cast<unsigned long>(shm_dev_)
    );
    return 0;
}

// ---------------------------------------------------------------------------
// Record accumulation (in-memory)
// ---------------------------------------------------------------------------

void ScopeStatsCollector::begin_run() {
    {
        std::scoped_lock lock(records_mutex_);
        const size_t charged = records_.release();
        if (charged > 0) host_budget_.credit(charged);
    }
    total_collected_ = 0;
    refused_records_ = 0;
    recovered_current_buf_ = 0;
    recovered_current_total_ = 0;
    execution_complete_.store(false, std::memory_order_release);

    if (shm_host_ == nullptr) return;

    // The device's record counters are producer-side and never reset by it, so
    // they carry the previous run's totals into this run's reconcile. The old
    // finalize/init cycle cleared them implicitly by memsetting a fresh region.
    // The two are adjacent, so one write-back covers both and leaves the
    // device-owned fields in the same line alone.
    ScopeStatsBufferState *state = get_scope_stats_buffer_state(shm_host_, 0);
    state->dropped_record_count = 0;
    state->total_record_count = 0;
    wmb();
    static_assert(
        offsetof(ScopeStatsBufferState, total_record_count) ==
            offsetof(ScopeStatsBufferState, dropped_record_count) + sizeof(uint32_t),
        "the two counters must stay adjacent for this single write-back to cover both"
    );
    publish_field(&state->dropped_record_count, 2 * sizeof(uint32_t), "record counters");
}

void ScopeStatsCollector::on_buffer_collected(const ScopeStatsReadyBufferInfo &info) {
    // The collector thread's entry point has no exception boundary of its own
    // — `ProfilerBase::consume` calls this directly — so one is established
    // here. A host failure below becomes a reported, persistent diagnostic
    // error rather than an escape that terminates the chip subprocess.
    try {
        append_buffer_records(info.host_buffer_ptr);
    } catch (...) {
        note_host_failure("a collected buffer could not be taken into host storage");
    }
}

void ScopeStatsCollector::append_buffer_records(const void *buf_host_ptr) {
    if (throw_in_collector_) throw std::bad_alloc();
    const ScopeStatsBuffer *buf = reinterpret_cast<const ScopeStatsBuffer *>(buf_host_ptr);
    uint32_t n = buf->count;
    if (n > static_cast<uint32_t>(PLATFORM_SCOPE_STATS_RECORDS_PER_BUFFER)) {
        n = static_cast<uint32_t>(PLATFORM_SCOPE_STATS_RECORDS_PER_BUFFER);
    }
    if (n == 0) return;

    std::scoped_lock lock(records_mutex_);
    // Identity comes from the same 64-byte header copy that carried `count`,
    // and is stored per record so it survives the device buffer going back to
    // the pool and being re-stamped by a later run.
    const uint64_t run_epoch = buf->run_epoch;
    const uint32_t local_seq = buf->local_seq;
    for (uint32_t i = 0; i < n; i++) {
        const CollectedScopeStatsRecord record{buf->records[i], run_epoch, local_seq, 0};
        const bool stored = records_.append(
            record,
            [this](size_t bytes) {
                return charge_record_block(bytes);
            },
            [this](size_t bytes) {
                host_budget_.credit(bytes);
            }
        );
        if (!stored) {
            // Counted as received and not retained: the difference is the
            // loss this run's artifact reports. The receive path is never
            // blocked and nothing is allocated on this branch.
            refused_records_ += static_cast<uint64_t>(n - i);
            total_collected_ += n;
            return;
        }
    }
    total_collected_ += n;
}

std::vector<CollectedScopeStatsRecord> ScopeStatsCollector::collected_records() const {
    std::scoped_lock lock(records_mutex_);
    std::vector<CollectedScopeStatsRecord> out;
    out.reserve(records_.size());
    for (size_t i = 0; i < records_.size(); i++)
        out.push_back(records_[i]);
    return out;
}

size_t ScopeStatsCollector::collected_for_run(uint64_t run_epoch) const {
    std::scoped_lock lock(records_mutex_);
    size_t n = 0;
    for (size_t i = 0; i < records_.size(); i++) {
        if (records_[i].run_epoch == run_epoch) n++;
    }
    return n;
}

// ---------------------------------------------------------------------------
// reconcile_counters
// ---------------------------------------------------------------------------

bool ScopeStatsCollector::reconcile_counters() {
    if (shm_host_ == nullptr) return false;
    report_drain_drops();

    // Pull the latest BufferState (current_buf_ptr, total/dropped counters)
    // before the cross-check so it sees post-stop() device state.
    if (manager_.shared_mem_dev() != nullptr && shm_size_ > 0) {
        profiling_copy_from_device(shm_host_, manager_.shared_mem_dev(), shm_size_);
    }
    rmb();
    bool clean = true;

    ScopeStatsBufferState *state = scope_stats_state(0);
    uint64_t total_device = state->total_record_count;
    uint64_t dropped_device = state->dropped_record_count;

    uint64_t buf_dev = state->current_buf_ptr;
    if (buf_dev != 0) {
        void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_dev));
        if (host_ptr != nullptr) {
            profiling_copy_from_device(host_ptr, reinterpret_cast<void *>(buf_dev), sizeof(ScopeStatsBuffer));
            uint32_t count = reinterpret_cast<const ScopeStatsBuffer *>(host_ptr)->count;
            if (count != 0) {
                if (recovered_current_buf_ != buf_dev || recovered_current_total_ != total_device) {
                    append_buffer_records(host_ptr);
                    recovered_current_buf_ = buf_dev;
                    recovered_current_total_ = total_device;
                    LOG_WARN(
                        "scope_stats reconcile: recovered un-flushed buffer "
                        "(current_buf_ptr=0x%lx, count=%u) host-side; device flush did not run",
                        static_cast<unsigned long>(buf_dev), count
                    );
                } else {
                    LOG_WARN(
                        "scope_stats reconcile: un-flushed buffer "
                        "(current_buf_ptr=0x%lx, count=%u) was already recovered host-side",
                        static_cast<unsigned long>(buf_dev), count
                    );
                }
                clean = false;
            }
        } else {
            LOG_ERROR(
                "scope_stats reconcile: un-flushed buffer current_buf_ptr=0x%lx has no host mapping",
                static_cast<unsigned long>(buf_dev)
            );
            clean = false;
        }
    }

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
        LOG_INFO(
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

namespace {

using simpler::dfx::scope_stats_runs::Collection;
using simpler::dfx::scope_stats_runs::DeviceSnapshot;
using simpler::dfx::scope_stats_runs::RecordBlocks;

/**
 * Bounded staging for the record lines.
 *
 * The bytes written are exactly what one growing `std::string` produced — the
 * per-record `snprintf` is unchanged — but the scratch no longer scales with
 * the record count, so it fits inside a fixed budget reservation.
 */
class LineSink {
public:
    explicit LineSink(std::FILE *fp) :
        fp_(fp) {
        stage_.reserve(simpler::dfx::runs::kWriterScratchBytes);
    }
    ~LineSink() { flush(); }

    void append(const char *data, size_t n) {
        if (stage_.size() + n > simpler::dfx::runs::kWriterScratchBytes) flush();
        if (n > simpler::dfx::runs::kWriterScratchBytes) {
            if (std::fwrite(data, 1, n, fp_) != n) ok_ = false;
            return;
        }
        stage_.append(data, n);
    }

    void flush() {
        if (stage_.empty()) return;
        if (std::fwrite(stage_.data(), 1, stage_.size(), fp_) != stage_.size()) ok_ = false;
        stage_.clear();
    }

    bool ok() const { return ok_; }

private:
    std::FILE *fp_;
    std::string stage_;
    bool ok_{true};
};

}  // namespace

simpler::dfx::scope_stats_runs::DeviceSnapshot ScopeStatsCollector::snapshot_unchecked() const {
    DeviceSnapshot out;
    if (shm_host_ == nullptr) return out;
    const ScopeStatsDataHeader *hdr = scope_stats_header();
    const ScopeStatsBufferState *state = scope_stats_state(0);
    out.fatal_latched = hdr->fatal_latched;
    out.dropped_records = state->dropped_record_count;
    out.total_records = state->total_record_count;
    for (int r = 0; r < SCOPE_STATS_MAX_RING_DEPTH; r++) {
        out.task_window_cap[r] = hdr->task_window_cap[r];
        out.dep_pool_cap[r] = hdr->dep_pool_cap[r];
        out.heap_cap[r] = hdr->heap_cap[r];
    }
    out.tensormap_cap = hdr->tensormap_cap;
    out.valid = true;
    return out;
}

int ScopeStatsCollector::render_jsonl_to(
    std::FILE *fp, const DeviceSnapshot &device, const RecordBlocks &records, const Collection *extra
) {
    // Line 1: run metadata. Per-ring capacities and the tensormap capacity are
    // run-constants, so they live here once rather than on every record.
    std::string task_window_max;
    std::string heap_max;
    std::string dep_pool_max;
    for (int r = 0; r < SCOPE_STATS_MAX_RING_DEPTH; r++) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%s%d", r == 0 ? "" : ", ", device.task_window_cap[r]);
        task_window_max += buf;
        std::snprintf(buf, sizeof(buf), "%s%" PRIu64, r == 0 ? "" : ", ", device.heap_cap[r]);
        heap_max += buf;
        std::snprintf(buf, sizeof(buf), "%s%d", r == 0 ? "" : ", ", device.dep_pool_cap[r]);
        dep_pool_max += buf;
    }
    // heap_start/heap_end are monotonic cumulative bytes, not wrapping ring
    // offsets — see docs/dfx/scope-stats.md.
    if (std::fprintf(
            fp,
            "{\"fatal\": %s, \"dropped\": %u, \"total\": %u, "
            "\"task_window_max\": [%s], \"heap_max\": [%s], \"dep_pool_max\": [%s], \"tensormap_max\": %d",
            device.fatal_latched ? "true" : "false", device.dropped_records, device.total_records,
            task_window_max.c_str(), heap_max.c_str(), dep_pool_max.c_str(), device.tensormap_cap
        ) < 0) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // Additive, and only in background mode: a default-path artifact keeps the
    // seven keys it has today, in the same order.
    if (extra != nullptr &&
        std::fprintf(
            fp,
            ", \"collection_verdict\": \"%s\", \"counts_unknown\": %s, \"host_received_records\": %llu, "
            "\"host_retained_records\": %llu",
            simpler::dfx::runs::verdict_name(extra->verdict), extra->counts_unknown ? "true" : "false",
            static_cast<unsigned long long>(extra->received), static_cast<unsigned long long>(extra->retained)
        ) < 0) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (std::fprintf(fp, "}\n") < 0) return PTO_RUNTIME_ERR_INTERNAL;

    LineSink sink(fp);
    char line[640];
    for (size_t i = 0; i < records.size(); i++) {
        const CollectedScopeStatsRecord &collected = records[i];
        const ScopeStatsRecord &rec = collected.record;
        const int site_len = static_cast<int>(strnlen(rec.site_file_basename, sizeof(rec.site_file_basename)));
        const char *phase = (rec.phase == SCOPE_STATS_PHASE_BEGIN) ? "begin" : "end";
        int n = std::snprintf(
            line, sizeof(line),
            "{\"site\": \"%.*s:%d\", \"phase\": \"%s\", \"depth\": %d, \"ring\": %d, "
            "\"task_window_start\": %d, \"task_window_end\": %d, "
            "\"heap_start\": %" PRIu64 ", \"heap_end\": %" PRIu64 ", "
            "\"dep_pool_start\": %d, \"dep_pool_end\": %d, "
            "\"tensormap\": %d, \"run_epoch\": %" PRIu64 ", \"buf_seq\": %u}\n",
            site_len, rec.site_file_basename, rec.site_line, phase, rec.depth, rec.ring_id, rec.task_start,
            rec.task_end, rec.heap_start, rec.heap_end, rec.dep_pool_start, rec.dep_pool_end, rec.tensormap_used,
            collected.run_epoch, collected.local_seq
        );
        if (n > 0) sink.append(line, static_cast<size_t>(n < static_cast<int>(sizeof(line)) ? n : sizeof(line) - 1));
    }
    sink.flush();
    return sink.ok() ? 0 : PTO_RUNTIME_ERR_INTERNAL;
}

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
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    std::scoped_lock lock(records_mutex_);
    const int rc = render_jsonl_to(fp, snapshot_unchecked(), records_, nullptr);
    std::fclose(fp);

    LOG_INFO(
        "scope_stats: wrote %lu records (dropped=%u) to %s", static_cast<unsigned long>(records_.size()),
        scope_stats_state(0)->dropped_record_count, path.c_str()
    );
    return rc;
}

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

void ScopeStatsCollector::finalize(ScopeStatsUnregisterCallback unregister_cb, const ScopeStatsFreeCallback &free_cb) {
    if (!initialized_) return;

    // Order matters and differs from the other retained collectors': the
    // quarantined copies may only be freed once the collector threads that
    // could still be appending to them are joined, and `stop()` is that join.
    finish_retained_runs();
    stop();
    discard_quarantined_runs();
    stop_writer();

    {
        std::scoped_lock lock(records_mutex_);
        (void)records_.release();
    }
    recovered_current_buf_ = 0;
    recovered_current_total_ = 0;

    auto release_dev = [&](void *p) {
        release_one_buffer(p, unregister_cb, free_cb);
    };

    // Free buffers still parked in the free_queue / current_buf_ptr. Release
    // the device pointer only — the paired host shadow stays in dev_to_host_
    // and is freed by clear_mappings() below (single source of truth for
    // shadow lifetime, no double-free).
    if (shm_host_ != nullptr) {
        ScopeStatsBufferState *state = scope_stats_state(0);
        release_dev(reinterpret_cast<void *>(state->current_buf_ptr));
        state->current_buf_ptr = 0;
        rmb();
        uint32_t head = state->free_queue.head;
        uint32_t tail = state->free_queue.tail;
        uint32_t queued = tail - head;
        if (queued > PLATFORM_SCOPE_STATS_SLOT_COUNT) queued = PLATFORM_SCOPE_STATS_SLOT_COUNT;
        for (uint32_t i = 0; i < queued; i++) {
            uint32_t slot = (head + i) % PLATFORM_SCOPE_STATS_SLOT_COUNT;
            release_dev(reinterpret_cast<void *>(state->free_queue.buffer_ptrs[slot]));
            state->free_queue.buffer_ptrs[slot] = 0;
        }
        state->free_queue.head = tail;
    }

    // Release framework-owned device allocations (recycled pool,
    // ready_queue, done_queue). Host shadows are freed by clear_mappings().
    manager_.release_owned_buffers([&](void *p) {
        release_dev(p);
    });

    // Free shared header region (device only — shadow stays in dev_to_host_
    // until clear_mappings).
    if (shm_dev_ != nullptr) {
        release_dev(shm_dev_);
        shm_dev_ = nullptr;
    }

    // Free remaining host shadows (per-state buffers + shm region).
    manager_.clear_mappings();

    initialized_ = false;
    total_collected_ = 0;
    clear_memory_context();
    LOG_INFO("ScopeStats collector finalized");
}
