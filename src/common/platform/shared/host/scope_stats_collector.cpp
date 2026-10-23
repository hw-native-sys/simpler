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

#include <algorithm>
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
    const ScopeStatsFreeCallback &free_cb, int device_id, ScopeStatsUnregisterCallback unregister_cb
) try {
    if (register_cb != nullptr && unregister_cb == nullptr) {
        LOG_ERROR("ScopeStats collector registration requires an unregister callback");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    if (terminal_banks_.backing_retained()) {
        LOG_ERROR("ScopeStatsCollector: terminal backing still requires cleanup before initialization");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
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

    device_alloc_cb_ = alloc_cb;
    device_free_cb_ = free_cb;
    bank_free_cb_ = free_cb;
    total_collected_ = 0;
    refused_records_ = 0;
    expected_producer_epoch_ = 0;
    device_snapshot_ = {};
    ordinary_capacities_ = {};
    ordinary_stop_unproved_ = false;
    ordinary_run_epoch_.reset();
    (void)records_.release();
    recovered_current_buf_ = 0;
    recovered_current_total_ = 0;
    execution_complete_.store(false, std::memory_order_release);

    // Stash callbacks on the base up-front so alloc_paired_buffer sees
    // consistent values during init. shm_host_ stays nullptr until the shm
    // allocation succeeds — start(tf) gates on shm_host_.
    set_memory_context(
        alloc_cb, register_cb, free_cb, profiling_copy_to_device_or_null(), profiling_copy_from_device_or_null(),
        /*shm_dev=*/nullptr, /*shm_host=*/nullptr, /*shm_size=*/0, device_id, unregister_cb
    );

    // RAII rollback: any early return after this point releases every
    // framework-tracked buffer (shm region + per-buffer-state PmuBuffer-style
    // entries) via free_cb. `guard.commit()` runs on the success path before
    // the trailing return 0.
    profiling_common::InitRollbackGuard<decltype(manager_)> guard(
        manager_, free_cb, register_cb != nullptr ? unregister_cb : nullptr, device_id
    );

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
    if (int rc = profiling_copy_to_device(shm_dev_local, shm_host_local, shm_size); rc != 0) return rc;

    guard.commit();
    shm_dev_ = shm_dev_local;

    publish_shared_memory(shm_dev_local, shm_host_local, shm_size);
    initialized_ = true;

    LOG_INFO(
        "ScopeStats collector initialized: %d threads, SHM=0x%lx", num_threads,
        reinterpret_cast<unsigned long>(shm_dev_)
    );
    return 0;
} catch (const std::bad_alloc &) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

// ---------------------------------------------------------------------------
// Record accumulation (in-memory)
// ---------------------------------------------------------------------------

void ScopeStatsCollector::begin_run(
    uint64_t run_epoch, const simpler::dfx::scope_stats_runs::CapacityMetadata *capacities
) {
    expected_producer_epoch_ = capacities == nullptr || capacities->required ? run_epoch : 0;
    device_snapshot_ = {};
    ordinary_capacities_ = capacities == nullptr ? simpler::dfx::scope_stats_runs::CapacityMetadata{} : *capacities;
    ordinary_run_epoch_ = run_epoch;
    ordinary_stop_unproved_ = false;
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

    // An unproven close can leave publications queued. A restarted ordinary
    // collector may retire that transport entry but cannot adopt its records.
    if (!terminal_banks_.opened() && ordinary_run_epoch_ && run_epoch != *ordinary_run_epoch_) {
        foreign_epoch_records_ += n;
        if (!foreign_epoch_reported_) {
            LOG_ERROR(
                "scope_stats: %u records stamped 0x%llx do not belong to ordinary run 0x%llx; not exported", n,
                static_cast<unsigned long long>(run_epoch), static_cast<unsigned long long>(*ordinary_run_epoch_)
            );
            foreign_epoch_reported_ = true;
            foreign_epoch_pending_report_ = true;
        }
        return;
    }

    // Which run's receipts these records belong to. With a bank mapped the
    // answer is the identity the producer stamped, because more than one run
    // can be open: appending to a single accumulator would put a successor's
    // records in its predecessor's artifact and count both against whichever
    // was admitted last. Without one there is only ever one open run, and the
    // accumulator is left exactly as it was so its artifact's bytes do not move.
    simpler::dfx::scope_stats_runs::RecordBlocks *store = &records_;
    uint64_t *received = &total_collected_;
    uint64_t *refused = &refused_records_;
    if (terminal_banks_.opened()) {
        RunReceipts *owner = nullptr;
        for (RunReceipts &entry : run_receipts_) {
            if (entry.open && entry.run_epoch == run_epoch) {
                owner = &entry;
                break;
            }
        }
        if (owner == nullptr) {
            // A run this collector holds no open receipts for: already sealed,
            // or never admitted. Counted against no run — adding it to one that
            // does not own it is the misattribution this routing exists to stop
            // — and reported, because a count nothing reads is not a loss
            // anybody learns about. Sticky and one-shot: the condition is a
            // collector-level ownership failure, and the receive path must not
            // log per record.
            foreign_epoch_records_ += n;
            total_collected_ += n;
            const bool first = !foreign_epoch_reported_;
            foreign_epoch_reported_ = true;
            if (first) {
                LOG_ERROR(
                    "scope_stats: %u records arrived stamped 0x%llx, which names no admitted run; they belong to no "
                    "artifact and this collector reports the loss",
                    n, static_cast<unsigned long long>(run_epoch)
                );
                // Recorded outside this lock by the caller's own path: a sticky
                // host failure is the collector's error channel and takes
                // `retained_mu_`, which the receive path must not take here.
                foreign_epoch_pending_report_ = true;
            }
            return;
        }
        store = &owner->records;
        received = &owner->received;
        refused = &owner->refused;
    }

    for (uint32_t i = 0; i < n; i++) {
        const CollectedScopeStatsRecord record{buf->records[i], run_epoch, local_seq, 0};
        const bool stored = store->append(
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
            *refused += static_cast<uint64_t>(n - i);
            *received += n;
            if (received != &total_collected_) total_collected_ += n;
            return;
        }
    }
    *received += n;
    // The collector-wide arrival count stays truthful on both paths: it is this
    // collector's telemetry — what `total_collected()` reports and what a caller
    // waits on — and is deliberately not what a run's verdict is built from,
    // which is the per-run receipts above. Skipped when they are the same
    // counter, on the unmapped path.
    if (received != &total_collected_) total_collected_ += n;
}

bool ScopeStatsCollector::open_run_receipts(uint64_t run_epoch) {
    std::scoped_lock lock(records_mutex_);
    for (const RunReceipts &entry : run_receipts_) {
        // One identity, one entry: a repeated admission of the same run would
        // otherwise split its records across two of them.
        if (entry.open && entry.run_epoch == run_epoch) return false;
    }
    for (RunReceipts &entry : run_receipts_) {
        if (entry.open) continue;
        const size_t charged = entry.records.release();
        if (charged > 0) host_budget_.credit(charged);
        entry.run_epoch = run_epoch;
        entry.received = 0;
        entry.refused = 0;
        entry.open = true;
        return true;
    }
    return false;
}

bool ScopeStatsCollector::take_run_receipts(uint64_t run_epoch, simpler::dfx::scope_stats_runs::RunExport *into) {
    std::scoped_lock lock(records_mutex_);
    for (RunReceipts &entry : run_receipts_) {
        if (!entry.open || entry.run_epoch != run_epoch) continue;
        into->records = std::move(entry.records);
        into->collection.received = entry.received;
        into->collection.retained = entry.received - entry.refused;
        entry.records = simpler::dfx::scope_stats_runs::RecordBlocks{};
        entry.open = false;
        entry.run_epoch = 0;
        return true;
    }
    return false;
}

void ScopeStatsCollector::drop_run_receipts(uint64_t run_epoch) {
    size_t charged = 0;
    {
        std::scoped_lock lock(records_mutex_);
        for (RunReceipts &entry : run_receipts_) {
            if (!entry.open || entry.run_epoch != run_epoch) continue;
            charged = entry.records.release();
            entry.open = false;
            entry.run_epoch = 0;
            entry.received = 0;
            entry.refused = 0;
            break;
        }
    }
    if (charged > 0) {
        std::lock_guard<std::mutex> lk(retained_mu_);
        host_budget_.credit(charged);
    }
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
// Per-run terminal banks
// ---------------------------------------------------------------------------

void *ScopeStatsCollector::bank_alloc(void *owner, size_t bytes, void **host_out) {
    auto *self = static_cast<ScopeStatsCollector *>(owner);
    // Paired, because an allocation's host readability is a property of how it
    // was mapped and not of the platform: the raw device allocator alone leaves
    // an address this process may not dereference. Kept out of the buffer
    // pool's mapping table, whose teardown would free it on a path that proves
    // nothing about this collector's producers.
    return self->alloc_unmanaged_paired_buffer(bytes, host_out);
}

int ScopeStatsCollector::bank_unmap(void *owner, void *device_ptr) {
    auto *self = static_cast<ScopeStatsCollector *>(owner);
    return self->unmap_unmanaged_paired_buffer(device_ptr, self->bank_unregister_cb_);
}

int ScopeStatsCollector::bank_free(void *owner, void *device_ptr) {
    auto *self = static_cast<ScopeStatsCollector *>(owner);
    if (self->bank_free_cb_ == nullptr || device_ptr == nullptr) return -1;
    // The allocator's own status, reported rather than dropped: a refused free
    // may leave the allocator still owning the pointer, and the bank array is
    // the memory a producer writes.
    return self->bank_free_cb_(device_ptr);
}

int ScopeStatsCollector::bank_copy(
    void *owner, void *host_dst, const void *device_src, const void *host_src, size_t bytes
) {
    auto *self = static_cast<ScopeStatsCollector *>(owner);
    // Both addresses and this collector's own copy context, so the shared rule
    // reads the mapping this allocation actually has rather than the one its
    // platform usually gives.
    return simpler::dfx::credit::read_run_terminal_bank(host_dst, device_src, host_src, bytes, self->copy_from_device_);
}

int ScopeStatsCollector::bank_initialize(
    void *owner, void *device_dst, void *host_dst, const void *initial, size_t bytes
) {
    auto *self = static_cast<ScopeStatsCollector *>(owner);
    return simpler::dfx::credit::initialize_run_terminal_bank(
        device_dst, host_dst, initial, bytes, self->copy_to_device_
    );
}

bool ScopeStatsCollector::open_run_terminal_banks(int expected_producers) {
    if (!retain_across_runs_) return false;
    if (shm_host_ == nullptr || device_alloc_cb_ == nullptr || device_free_cb_ == nullptr) return false;
    if (expected_producers != SCOPE_STATS_TERMINAL_PRODUCERS) return false;
    return terminal_banks_.open(
        SCOPE_STATS_TERMINAL_PRODUCERS, expected_producers, this, &ScopeStatsCollector::bank_alloc,
        &ScopeStatsCollector::bank_unmap, &ScopeStatsCollector::bank_free, &ScopeStatsCollector::bank_copy,
        &ScopeStatsCollector::bank_initialize
    );
}

// ---------------------------------------------------------------------------
// reconcile_counters
// ---------------------------------------------------------------------------

int ScopeStatsCollector::close_ordinary_run(const std::string &output_dir, bool producer_stop_proved) {
    if (producer_stop_proved) {
        ordinary_stop_unproved_ = false;
        quiesce();
        reconcile_counters();
    } else {
        // Do not chase live publications or recover the device's held buffer.
        // stop(false) still finishes already-handed host entries and joins them.
        stop(false);
        report_drain_drops();
        device_snapshot_ = {};
        ordinary_stop_unproved_ = true;
    }
    report_foreign_epoch_records();
    return write_jsonl(output_dir);
}

bool ScopeStatsCollector::reconcile_counters() {
    if (shm_host_ == nullptr) return false;
    report_drain_drops();

    simpler::dfx::scope_stats_runs::DeviceSnapshot device;
    if (!read_terminal_checked(&device)) return false;
    bool clean = true;

    ScopeStatsBufferState *state = scope_stats_state(0);
    uint64_t total_device = device.total_records;
    uint64_t dropped_device = device.dropped_records;

    uint64_t buf_dev = state->current_buf_ptr;
    if (buf_dev != 0) {
        void *host_ptr = manager_.resolve_host_ptr(reinterpret_cast<void *>(buf_dev));
        if (host_ptr != nullptr) {
            if (profiling_copy_from_device(host_ptr, reinterpret_cast<void *>(buf_dev), sizeof(ScopeStatsBuffer)) !=
                0) {
                device_snapshot_ = {};
                LOG_ERROR("scope_stats: reading the held buffer failed; no artifact is written");
                return false;
            }
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
    if (expected_producer_epoch_ != 0 && state->run_epoch != expected_producer_epoch_) {
        LOG_ERROR(
            "scope_stats: producer epoch %llu does not match run %llu; no accounting is exported",
            static_cast<unsigned long long>(state->run_epoch), static_cast<unsigned long long>(expected_producer_epoch_)
        );
        return out;
    }
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
    std::FILE *fp, const DeviceSnapshot &device, const RecordBlocks &records, const Collection *extra,
    const simpler::dfx::scope_stats_runs::CapacityMetadata *host_only_capacities
) {
    DeviceSnapshot metadata = device;
    if (host_only_capacities != nullptr && host_only_capacities->values) {
        const auto &caps = *host_only_capacities->values;
        std::copy(caps.task_window.begin(), caps.task_window.end(), metadata.task_window_cap);
        std::copy(caps.heap.begin(), caps.heap.end(), metadata.heap_cap);
        std::copy(caps.dep_pool.begin(), caps.dep_pool.end(), metadata.dep_pool_cap);
        metadata.tensormap_cap = caps.tensormap;
    }
    // Unknown close has no final device accounting. Its capacities, if known,
    // come from the immutable host binding rather than the shared mirror.
    if (host_only_capacities != nullptr) {
        if (std::fprintf(fp, "{\"fatal\": null, \"dropped\": null, \"total\": null") < 0) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
    } else if (std::fprintf(
                   fp, "{\"fatal\": %s, \"dropped\": %u, \"total\": %u", metadata.fatal_latched ? "true" : "false",
                   metadata.dropped_records, metadata.total_records
               ) < 0) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (host_only_capacities != nullptr && !host_only_capacities->values) {
        if (std::fprintf(
                fp, ", \"task_window_max\": null, \"heap_max\": null, \"dep_pool_max\": null, \"tensormap_max\": null"
            ) < 0) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
    } else {
        std::string task_window_max;
        std::string heap_max;
        std::string dep_pool_max;
        for (int r = 0; r < SCOPE_STATS_MAX_RING_DEPTH; r++) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s%d", r == 0 ? "" : ", ", metadata.task_window_cap[r]);
            task_window_max += buf;
            std::snprintf(buf, sizeof(buf), "%s%" PRIu64, r == 0 ? "" : ", ", metadata.heap_cap[r]);
            heap_max += buf;
            std::snprintf(buf, sizeof(buf), "%s%d", r == 0 ? "" : ", ", metadata.dep_pool_cap[r]);
            dep_pool_max += buf;
        }
        if (std::fprintf(
                fp, ", \"task_window_max\": [%s], \"heap_max\": [%s], \"dep_pool_max\": [%s], \"tensormap_max\": %d",
                task_window_max.c_str(), heap_max.c_str(), dep_pool_max.c_str(), metadata.tensormap_cap
            ) < 0) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
    }
    // Retained exports and unproven ordinary closes report collection quality.
    // A proved ordinary export keeps its existing seven-key metadata line.
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
    if (!ordinary_stop_unproved_ && !device_snapshot_.valid) return PTO_RUNTIME_ERR_INVALID_STATE;

    std::scoped_lock lock(records_mutex_);
    const std::filesystem::path dir = std::filesystem::path(output_dir) / "scope_stats";
    const std::string path = (dir / "scope_stats.jsonl").string();
    std::error_code ec;
    // No host sample means no evidence for this run. A reused output prefix
    // must not leave the predecessor's healthy file at the current result path.
    if (ordinary_stop_unproved_ && records_.size() == 0) {
        std::filesystem::remove(path, ec);
        if (ec) {
            LOG_ERROR("scope_stats: failed to remove stale artifact %s: %s", path.c_str(), ec.message().c_str());
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        return 0;
    }

    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LOG_WARN("scope_stats: failed to create output dir %s: %s", dir.c_str(), ec.message().c_str());
    }
    std::FILE *fp = std::fopen(path.c_str(), "w");
    if (fp == nullptr) {
        LOG_ERROR("scope_stats: failed to open %s", path.c_str());
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // This verdict describes only the host collection; it settles no device references.
    const Collection unknown{simpler::dfx::runs::Verdict::CutUnknown, true, total_collected_, records_.size()};
    const int rc = render_jsonl_to(
        fp, device_snapshot_, records_, ordinary_stop_unproved_ ? &unknown : nullptr,
        ordinary_stop_unproved_ ? &ordinary_capacities_ : nullptr
    );
    std::fclose(fp);

    LOG_INFO(
        "scope_stats: wrote %lu host records (device accounting %s) to %s", static_cast<unsigned long>(records_.size()),
        ordinary_stop_unproved_ ? "unknown" : "checked", path.c_str()
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
        // Every per-run entry too: a run quarantined or still open at teardown
        // keeps its records in its own entry, and the reader threads are joined
        // by now, so this is the first point at which they are nobody's.
        for (RunReceipts &entry : run_receipts_) {
            (void)entry.records.release();
            entry = RunReceipts{};
        }
        foreign_epoch_records_ = 0;
    }
    recovered_current_buf_ = 0;
    recovered_current_total_ = 0;

    if (!validate_release_callbacks(unregister_cb, free_cb)) return;
    // Released through this teardown's own callback, the one that also frees
    // `state->current_buf_ptr` below — a buffer a producer writes into — so
    // the array follows exactly the policy the caller chose for every other
    // device resource here: freed on the ordinary paths, and abandoned
    // untouched on a teardown whose callback frees nothing. The
    // `initialize`-time allocator is deliberately not used: it would hand this
    // memory back on a path where the caller abandoned everything else.
    // A release the callback refuses leaves the array mapped and reported.
    bank_free_cb_ = free_cb;
    bank_unregister_cb_ = unregister_cb;
    const auto bank_release = terminal_banks_.release_backing(/*device_stopped=*/true);
    if (terminal_banks_.backing_retained()) {
        note_host_failure(
            bank_release == simpler::dfx::credit::RunTerminalBanks::Release::FreeFailed ?
                "the device refused this collector's per-run terminal bank array; it stays mapped" :
                "this collector's per-run terminal bank array could not be released and stays mapped"
        );
        return;
    }

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
